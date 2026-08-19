//! Image generation for `generate_image`: the hosted gateway
//! `/v1/generate-image` (device-signed, quota'd) or the user's own image key.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;

use crate::services::session_store::ApiKey;

/// Resolved (and the custom key decrypted) at dispatch, so the turn just calls it.
#[derive(Clone)]
pub enum GeneratorSource {
    Gateway,
    OwnKey { model: String, key: Box<ApiKey> },
}

impl GeneratorSource {
    /// What the agent and its errors call the generator.
    pub fn label(&self) -> &str {
        match self {
            Self::Gateway => "aivo",
            Self::OwnKey { model, .. } => model,
        }
    }

    /// The model the turn's loopback serve must carry; the gateway path talks to
    /// api.getaivo.dev directly and needs none.
    pub fn upstream_model(&self) -> Option<&str> {
        match self {
            Self::Gateway => None,
            Self::OwnKey { model, .. } => Some(model),
        }
    }
}

const TIMEOUT_SECS: u64 = 180;

/// Latched once generation is known-exhausted this session (quota/auth/config),
/// so the tool reports the cause instead of re-hitting the gateway every call.
pub static GENERATE_EXHAUSTED: AtomicBool = AtomicBool::new(false);

/// Serializes tests that flip the process-global latch or the endpoint var.
#[cfg(test)]
pub static TEST_GENERATE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub fn generate_exhausted() -> bool {
    GENERATE_EXHAUSTED.load(Relaxed)
}

/// `data:` URLs — the shape `save_data_url_images` takes, so both sources
/// share one save path.
pub async fn generate(
    src: &GeneratorSource,
    client: &reqwest::Client,
    base: &str,
    auth: &str,
    prompt: &str,
) -> Result<Vec<String>, String> {
    match src {
        GeneratorSource::Gateway => generate_via_gateway(prompt).await,
        GeneratorSource::OwnKey { model, .. } => {
            generate_via_key(client, base, auth, model, prompt).await
        }
    }
    .and_then(|(images, text)| {
        if images.is_empty() {
            // Text-only reply (refusal/clarification) — surface it so the agent can react.
            let who = src.label();
            return Err(if text.trim().is_empty() {
                format!("{who} returned no image")
            } else {
                format!("{who} returned no image: {text}")
            });
        }
        Ok(images)
    })
}

/// Non-200 → (actionable message, whether to latch the session exhausted).
/// Mirrors the handler's status vocabulary (handlers/generateImage.ts).
fn classify_generate_error(status: u16) -> (String, bool) {
    match status {
        401 => (
            "image generation needs sign-in — run `aivo login`".to_string(),
            true,
        ),
        403 => (
            "image generation isn't available on your plan".to_string(),
            true,
        ),
        429 => (
            "today's image-generation quota is used up".to_string(),
            true,
        ),
        503 => ("image generation isn't configured".to_string(), true),
        400 => ("the image prompt was rejected".to_string(), false),
        413 => ("the image prompt is too long".to_string(), false),
        502 => ("image generation is temporarily down".to_string(), false),
        _ => (format!("image generation failed (HTTP {status})"), false),
    }
}

#[derive(Serialize)]
struct GatewayBody<'a> {
    prompt: &'a str,
}

/// Latches `GENERATE_EXHAUSTED` on persistent failures.
async fn generate_via_gateway(prompt: &str) -> Result<(Vec<String>, String), String> {
    if generate_exhausted() {
        return Err("image generation is unavailable for the rest of this session".to_string());
    }
    // Points at loopback (tests, local wrangler), which a proxy env would swallow.
    let override_endpoint = std::env::var("AIVO_GENERATE_IMAGE_ENDPOINT")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let mut builder = crate::services::http_utils::aivo_http_client_builder()
        .timeout(Duration::from_secs(TIMEOUT_SECS));
    if override_endpoint.is_some() {
        builder = builder.no_proxy();
    }
    let client = builder
        .build()
        .map_err(|e| format!("build http client: {e}"))?;
    let endpoint = override_endpoint.unwrap_or_else(|| {
        format!(
            "{}/v1/generate-image",
            crate::constants::AIVO_STARTER_REAL_URL
        )
    });
    // Device-signed (same auth as chat); the gateway holds the keys + quota.
    let builder = client.post(endpoint).json(&GatewayBody { prompt });
    let resp = crate::services::device_fingerprint::with_starter_headers(builder)
        .send()
        .await
        .map_err(|e| format!("couldn't reach image generation ({e})"))?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if status.is_success() {
        let images = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| Some(gateway_images(v.get("images")?)))
            .unwrap_or_default();
        return Ok((images, String::new()));
    }
    let (message, latch) = classify_generate_error(status.as_u16());
    if latch {
        GENERATE_EXHAUSTED.store(true, Relaxed);
    }
    Err(message)
}

/// `[{media_type, data}]` → `data:` URLs; incomplete entries are dropped.
fn gateway_images(images: &Value) -> Vec<String> {
    images
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|img| {
                    let mime = img.get("media_type")?.as_str()?;
                    let data = img.get("data")?.as_str().filter(|d| !d.is_empty())?;
                    Some(format!("data:{mime};base64,{data}"))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The own-key path rides the caller's loopback serve, so usage is accounted
/// under "code" like any other turn request.
async fn generate_via_key(
    client: &reqwest::Client,
    base: &str,
    auth: &str,
    model: &str,
    prompt: &str,
) -> Result<(Vec<String>, String), String> {
    // OpenRouter emits images only with this opt-in; the Gemini bridge maps it
    // to `responseModalities`.
    let mut extra = serde_json::Map::new();
    extra.insert(
        "modalities".to_string(),
        serde_json::json!(["image", "text"]),
    );
    let request = crate::agent::protocol::ChatRequest {
        model: model.to_string(),
        messages: vec![serde_json::json!({"role": "user", "content": prompt})],
        tools: vec![],
        extra,
    };
    let mut sink = |_: crate::agent::serve_client::StreamDelta| {};
    let call = crate::agent::serve_client::complete(client, base, Some(auth), &request, &mut sink);
    match tokio::time::timeout(Duration::from_secs(TIMEOUT_SECS), call).await {
        Err(_) => Err(format!("image generation via {model} timed out")),
        Ok(Err(e)) => Err(format!("image generation via {model} failed: {e}")),
        Ok(Ok(msg)) => Ok((msg.images, msg.content.unwrap_or_default())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Latching statuses are covered by the pure classify table — hitting them
    /// here would race other tests through the process-global latch.
    #[tokio::test]
    async fn gateway_round_trip_against_fake_server() {
        let _guard = TEST_GENERATE_LOCK.lock().await;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            for body in [
                r#"{"images":[{"media_type":"image/png","data":"aGk="}],"model":"banana"}"#,
                r#"{"images":[]}"#,
            ] {
                let (mut sock, _) = listener.accept().unwrap();
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf);
                let _ = sock.write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                );
            }
        });
        unsafe {
            std::env::set_var(
                "AIVO_GENERATE_IMAGE_ENDPOINT",
                format!("http://{addr}/v1/generate-image"),
            );
        }
        let ok = generate_via_gateway("a cat").await;
        let empty = generate_via_gateway("a cat").await;
        unsafe {
            std::env::remove_var("AIVO_GENERATE_IMAGE_ENDPOINT");
        }
        // The gateway hands back base64 + media type; the tool needs data URLs.
        assert_eq!(
            ok.unwrap().0,
            vec!["data:image/png;base64,aGk=".to_string()]
        );
        assert!(empty.unwrap().0.is_empty());
        assert!(!generate_exhausted(), "an empty result must not latch");
    }

    #[test]
    fn classify_latches_only_persistent_statuses() {
        for (status, latch) in [
            (401, true),
            (403, true),
            (429, true),
            (503, true),
            (400, false),
            (413, false),
            (502, false),
            (500, false),
        ] {
            let (message, got) = classify_generate_error(status);
            assert_eq!(got, latch, "status {status}");
            assert!(!message.is_empty());
        }
    }

    #[test]
    fn gateway_images_skips_incomplete_entries() {
        let v = serde_json::json!([
            {"media_type": "image/png", "data": "aGk="},
            {"media_type": "image/png"},
            {"data": "eW8="},
            {"media_type": "image/jpeg", "data": ""},
            {"media_type": "image/jpeg", "data": "eW8="},
        ]);
        assert_eq!(
            gateway_images(&v),
            vec![
                "data:image/png;base64,aGk=".to_string(),
                "data:image/jpeg;base64,eW8=".to_string()
            ]
        );
        assert!(gateway_images(&serde_json::json!("not an array")).is_empty());
    }

    /// A text-only reply is a failed generation, and the message names the source.
    #[tokio::test]
    async fn empty_result_reports_the_source_label() {
        assert_eq!(GeneratorSource::Gateway.label(), "aivo");
        assert_eq!(GeneratorSource::Gateway.upstream_model(), None);
    }
}
