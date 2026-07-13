//! JoyCode Router — transforms OpenAI-format requests into JoyCode API format.
//!
//! JoyCode's API is OpenAI-compatible at the wire level but requires:
//! - Custom headers (`ptKey`, `source-type`, `loginType`)
//! - Request body wrapping (tenant, userId, client, clientVersion, language)
//! - Color gateway HMAC-SHA256 signing for enterprise tenants
//! - Special endpoint path mapping
//!
//! This router plugs into `serve_upstream` as a custom sender, analogous to
//! `send_anthropic_chat` / `send_gemini_chat`.

use anyhow::{Context, Result};
use hmac::{Hmac, Mac};
use serde_json::{Value, json};
use sha2::Sha256;

use crate::services::serve_upstream::{RouterResponse, StreamingBody, UpstreamRequestContext};

/// Color gateway app ID (reverse-engineered from JoyCode 2.7.5)
const COLOR_GATEWAY_APP_ID: &str = "joycode_ide";
/// Color gateway HMAC key
const COLOR_GATEWAY_HMAC_KEY: &[u8] = b"0691a3f0b37b4a85aeb63ad0fc7db3ed";
/// Color gateway path prefix
const COLOR_GATEWAY_PATH: &str = "/api";
/// JoyCode client version
const JOYCODE_CLIENT_VERSION: &str = "2.7.5";
/// JoyCode User-Agent
const JOYCODE_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
    AppleWebKit/537.36 (KHTML, like Gecko) \
    JoyCode/2.7.5 Chrome/133.0.0.0 Electron/35.2.0 Safari/537.36";

type HmacSha256 = Hmac<Sha256>;

/// Mapping from standard OpenAI endpoint paths to JoyCode color gateway
/// (functionId, v2 path) pairs.
struct ColorEndpoint {
    function_id: &'static str,
    v2_path: &'static str,
}

fn color_endpoints() -> HashMap<&'static str, ColorEndpoint> {
    use std::collections::HashMap;
    let mut m = HashMap::new();
    m.insert(
        "/api/saas/openai/v1/chat/completions",
        ColorEndpoint {
            function_id: "chat_completions",
            v2_path: "/api/saas/openai/v2/chat/completions",
        },
    );
    m.insert(
        "/api/saas/models/v1/modelList",
        ColorEndpoint {
            function_id: "joycode_modelList",
            v2_path: "/api/saas/models/v2/modelList",
        },
    );
    m.insert(
        "/api/saas/openai/v1/web-search",
        ColorEndpoint {
            function_id: "web_search",
            v2_path: "/api/saas/openai/v2/web-search",
        },
    );
    m.insert(
        "/api/saas/user/v1/userInfo",
        ColorEndpoint {
            function_id: "joycode_userInfo",
            v2_path: "/api/saas/user/v2/userInfo",
        },
    );
    m.insert(
        "/api/saas/anthropic/v1/messages",
        ColorEndpoint {
            function_id: "anthropic_completions",
            v2_path: "/api/saas/anthropic/v1/messages",
        },
    );
    m
}

use std::collections::HashMap;

/// Compute color gateway HMAC signature.
/// Canonical string = appid + "&" + functionId + "&" + timestamp
fn color_sign(function_id: &str) -> (String, String) {
    let ts = chrono::Utc::now().timestamp_millis().to_string();
    let sign_str = format!("{COLOR_GATEWAY_APP_ID}&{function_id}&{ts}");

    let mut mac = HmacSha256::new_from_slice(COLOR_GATEWAY_HMAC_KEY).unwrap();
    mac.update(sign_str.as_bytes());
    let result = mac.finalize();
    let sign = hex::encode(result.into_bytes());

    let query = format!("appid={COLOR_GATEWAY_APP_ID}&functionId={function_id}&t={ts}");
    (query, sign)
}

/// Build the final request URL for a JoyCode endpoint.
/// If color_base_url is set (enterprise tenant), use color gateway with signing.
/// Otherwise, use direct v2 path.
fn request_url(endpoint: &str, color_base_url: &str, master_base_url: &str) -> String {
    let endpoints = color_endpoints();

    if let Some(ep) = endpoints.get(endpoint) {
        if !color_base_url.is_empty() {
            // Color gateway mode with HMAC signing
            if let Ok(parsed) = url::Url::parse(color_base_url) {
                if !parsed.host_str().unwrap_or("").is_empty() {
                    let (query, sign) = color_sign(ep.function_id);
                    return format!(
                        "{}://{}{}?{}&sign={}",
                        parsed.scheme(),
                        parsed.host_str().unwrap(),
                        COLOR_GATEWAY_PATH,
                        query,
                        sign
                    );
                }
            }
        }
        // Direct v2 mode
        let base = if master_base_url.is_empty() {
            "https://joycode-api.jd.com"
        } else {
            master_base_url.trim_end_matches('/')
        };
        format!("{}{}", base, ep.v2_path)
    } else {
        // Unknown endpoint — direct to base
        format!("https://joycode-api.jd.com{}", endpoint)
    }
}

/// JoyCode credentials extracted from the stored key.
/// Serialized as JSON in `ApiKey.key`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JoyCodeKeyData {
    pub pt_key: String,
    pub user_id: String,
    pub color_base_url: String,
    pub master_base_url: String,
    pub tenant: String,
    pub login_type: String,
    pub org_full_name: String,
}

impl JoyCodeKeyData {
    /// Parse from the JSON stored in an ApiKey's key field.
    pub fn from_key_json(json_str: &str) -> Result<Self> {
        // Try parsing as JoyCode key data
        if let Ok(data) = serde_json::from_str::<Self>(json_str) {
            return Ok(data);
        }
        // Fallback: treat as a raw ptKey with empty metadata
        if !json_str.is_empty() && !json_str.starts_with('{') {
            return Ok(Self {
                pt_key: json_str.to_string(),
                user_id: String::new(),
                color_base_url: String::new(),
                master_base_url: String::new(),
                tenant: "JOYCODE".to_string(),
                login_type: "N_PIN_PC".to_string(),
                org_full_name: String::new(),
            });
        }
        anyhow::bail!("invalid JoyCode key data")
    }
}

/// Build JoyCode-specific request headers.
fn joycode_headers(pt_key: &str, login_type: &str) -> reqwest::header::HeaderMap {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        "Content-Type",
        "application/json; charset=UTF-8".parse().unwrap(),
    );
    headers.insert("source-type", "joycoder-ide".parse().unwrap());
    headers.insert("ptKey", pt_key.parse().unwrap());
    headers.insert(
        "loginType",
        (if login_type.is_empty() {
            "N_PIN_PC"
        } else {
            login_type
        })
        .parse()
        .unwrap(),
    );
    headers.insert("User-Agent", JOYCODE_USER_AGENT.parse().unwrap());
    headers.insert("Accept", "*/*".parse().unwrap());
    headers.insert("Accept-Encoding", "gzip, deflate".parse().unwrap());
    headers.insert(
        "Accept-Language",
        "zh-CN,zh;q=0.9,en;q=0.8".parse().unwrap(),
    );
    headers
}

/// Build Anthropic-compatible headers for JoyCode.
fn joycode_anthropic_headers(pt_key: &str, login_type: &str) -> reqwest::header::HeaderMap {
    let mut headers = joycode_headers(pt_key, login_type);
    if login_type.is_empty() {
        headers.insert("loginType", "PIN_JD_CLOUD".parse().unwrap());
    }
    headers
}

/// Wrap an OpenAI-format request body with JoyCode metadata.
fn wrap_joycode_body(body: &mut Value, key_data: &JoyCodeKeyData) {
    let wrapper = json!({
        "tenant": if key_data.tenant.is_empty() { "JOYCODE" } else { &key_data.tenant },
        "orgFullName": &key_data.org_full_name,
        "userId": &key_data.user_id,
        "client": "JoyCode",
        "clientVersion": JOYCODE_CLIENT_VERSION,
        "language": "UNKNOWN",
    });

    // Merge wrapper fields into the body (body fields take precedence for model/messages)
    if let Some(body_obj) = body.as_object_mut() {
        for (k, v) in wrapper.as_object().unwrap() {
            body_obj.entry(k.clone()).or_insert(v.clone());
        }
    }
}

/// Send a chat completion request through the JoyCode router.
/// Transforms the standard OpenAI request into JoyCode format.
pub(crate) async fn send_joycode_chat(
    body: &mut Value,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    // Parse key data from the API key
    let key_data = JoyCodeKeyData::from_key_json(&context.upstream_api_key)?;

    // Wrap the request body with JoyCode metadata
    wrap_joycode_body(body, &key_data);

    // Determine the endpoint
    let endpoint = "/api/saas/openai/v1/chat/completions";
    let url = request_url(
        endpoint,
        &key_data.color_base_url,
        &key_data.master_base_url,
    );

    // Build headers
    let headers = joycode_headers(&key_data.pt_key, &key_data.login_type);

    let client = &context.client;
    let req_builder = client.post(&url).headers(headers).json(&*body);

    // Send the request
    let response = req_builder
        .send()
        .await
        .context("send JoyCode chat request")?;

    let status = response.status().as_u16();

    if status != 200 {
        let error_body = response.text().await.unwrap_or_default();
        let error_msg = if error_body.len() > 500 {
            format!("{}...", &error_body[..500])
        } else {
            error_body
        };
        anyhow::bail!("JoyCode API error {status}: {error_msg}");
    }

    // JoyCode returns OpenAI-compatible responses, so we can pipe them through
    if client_wants_stream && body["stream"].as_bool().unwrap_or(false) {
        Ok(RouterResponse::Streaming {
            status: 200,
            content_type: "text/event-stream".to_string(),
            body: Box::new(StreamingBody::Upstream(response)),
        })
    } else {
        let resp_body = response.bytes().await.context("read JoyCode response")?;
        // Decompress gzip if needed
        let resp_bytes = if resp_body.starts_with(&[0x1f, 0x8b]) {
            use std::io::Read;
            let mut decoder = flate2::read::GzDecoder::new(&resp_body[..]);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)?;
            decompressed
        } else {
            resp_body.to_vec()
        };
        Ok(RouterResponse::buffered(
            200,
            "application/json",
            resp_bytes,
        ))
    }
}

/// Send a models list request through the JoyCode router.
pub(crate) async fn send_joycode_models(
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    let key_data = JoyCodeKeyData::from_key_json(&context.upstream_api_key)?;

    let endpoint = "/api/saas/models/v1/modelList";
    let url = request_url(
        endpoint,
        &key_data.color_base_url,
        &key_data.master_base_url,
    );
    let headers = joycode_headers(&key_data.pt_key, &key_data.login_type);

    let body = json!({
        "tenant": if key_data.tenant.is_empty() { "JOYCODE" } else { &key_data.tenant },
        "orgFullName": &key_data.org_full_name,
        "userId": &key_data.user_id,
        "client": "JoyCode",
        "clientVersion": JOYCODE_CLIENT_VERSION,
        "language": "UNKNOWN",
    });

    let response = context
        .client
        .post(&url)
        .headers(headers)
        .json(&body)
        .send()
        .await
        .context("send JoyCode models request")?;

    let status = response.status().as_u16();
    if status != 200 {
        let error_body = response.text().await.unwrap_or_default();
        anyhow::bail!(
            "JoyCode models error {status}: {}",
            &error_body[..error_body.len().min(500)]
        );
    }

    let resp_body = response
        .bytes()
        .await
        .context("read JoyCode models response")?;
    let resp_bytes = if resp_body.starts_with(&[0x1f, 0x8b]) {
        use std::io::Read;
        let mut decoder = flate2::read::GzDecoder::new(&resp_body[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        decompressed
    } else {
        resp_body.to_vec()
    };

    // Translate JoyCode model list to OpenAI /v1/models format
    let jc_resp: Value = serde_json::from_slice(&resp_bytes).context("parse JoyCode models")?;
    let openai_models = translate_models(jc_resp);

    Ok(RouterResponse::buffered(
        200,
        "application/json",
        serde_json::to_vec(&openai_models)?,
    ))
}

/// Translate JoyCode model list response to OpenAI /v1/models format.
fn translate_models(jc_resp: Value) -> Value {
    let data = jc_resp["data"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|m| {
            let id = m["modelId"]
                .as_str()
                .or_else(|| m["label"].as_str())
                .unwrap_or("unknown")
                .to_string();
            json!({
                "id": id,
                "object": "model",
                "created": 1700000000,
                "owned_by": "joycode"
            })
        })
        .collect::<Vec<_>>();

    json!({
        "object": "list",
        "data": data
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_sign_produces_hex_signature() {
        let (query, sign) = color_sign("chat_completions");
        assert!(query.contains("appid=joycode_ide"));
        assert!(query.contains("functionId=chat_completions"));
        assert!(!sign.is_empty());
        // Sign should be a hex string (64 chars for SHA256)
        assert_eq!(sign.len(), 64);
        assert!(sign.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn request_url_direct_mode() {
        let url = request_url("/api/saas/openai/v1/chat/completions", "", "");
        assert_eq!(
            url,
            "https://joycode-api.jd.com/api/saas/openai/v2/chat/completions"
        );
    }

    #[test]
    fn request_url_custom_master() {
        let url = request_url(
            "/api/saas/openai/v1/chat/completions",
            "",
            "https://custom-joycode.example.com",
        );
        assert_eq!(
            url,
            "https://custom-joycode.example.com/api/saas/openai/v2/chat/completions"
        );
    }

    #[test]
    fn request_url_color_gateway() {
        let url = request_url(
            "/api/saas/openai/v1/chat/completions",
            "https://api-ai.jd.com",
            "",
        );
        assert!(url.starts_with("https://api-ai.jd.com/api?"));
        assert!(url.contains("appid=joycode_ide"));
        assert!(url.contains("functionId=chat_completions"));
        assert!(url.contains("&sign="));
    }

    #[test]
    fn wrap_joycode_body_adds_metadata() {
        let mut body = json!({
            "model": "JoyAI-Code",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        });

        let key_data = JoyCodeKeyData {
            pt_key: "test_key".to_string(),
            user_id: "user123".to_string(),
            color_base_url: String::new(),
            master_base_url: String::new(),
            tenant: "JOYCODE".to_string(),
            login_type: "N_PIN_PC".to_string(),
            org_full_name: String::new(),
        };

        wrap_joycode_body(&mut body, &key_data);

        assert_eq!(body["tenant"], "JOYCODE");
        assert_eq!(body["userId"], "user123");
        assert_eq!(body["client"], "JoyCode");
        assert_eq!(body["model"], "JoyAI-Code"); // body fields take precedence
    }

    #[test]
    fn joycode_key_data_parse_raw_key() {
        let data = JoyCodeKeyData::from_key_json("my-raw-pt-key").unwrap();
        assert_eq!(data.pt_key, "my-raw-pt-key");
        assert_eq!(data.tenant, "JOYCODE");
    }

    #[test]
    fn joycode_key_data_parse_json() {
        let json = r#"{"pt_key":"key123","user_id":"u1","color_base_url":"","master_base_url":"","tenant":"MY_TENANT","login_type":"PIN_JD_CLOUD","org_full_name":"MyOrg"}"#;
        let data = JoyCodeKeyData::from_key_json(json).unwrap();
        assert_eq!(data.pt_key, "key123");
        assert_eq!(data.tenant, "MY_TENANT");
        assert_eq!(data.login_type, "PIN_JD_CLOUD");
    }

    #[test]
    fn translate_models_converts_jc_to_openai() {
        let jc = json!({
            "data": [
                {"label": "JoyAI Code", "modelId": "JoyAI-Code"},
                {"label": "Claude", "modelId": "claude-sonnet-4-6"}
            ]
        });
        let result = translate_models(jc);
        assert_eq!(result["object"], "list");
        let data = result["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["id"], "JoyAI-Code");
        assert_eq!(data[0]["owned_by"], "joycode");
        assert_eq!(data[1]["id"], "claude-sonnet-4-6");
    }

    #[test]
    fn joycode_headers_include_pt_key() {
        let headers = joycode_headers("my_key_123", "N_PIN_PC");
        assert_eq!(headers.get("ptKey").unwrap(), "my_key_123");
        assert_eq!(headers.get("source-type").unwrap(), "joycoder-ide");
        assert_eq!(headers.get("loginType").unwrap(), "N_PIN_PC");
    }
}
