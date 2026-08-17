//! Minimal dependency-free JSON Schema validator backing `--json-schema`:
//! `type`/`enum`/`const`, object/array/string/number checks, combinators, and
//! local `$ref`. Unknown keywords (`pattern`, `format`, …) are ignored so a
//! fancy schema degrades to partial checking rather than false failures.

use serde_json::Value;

/// Parse the answer as JSON (tolerating one markdown fence) and validate it.
/// The error string is model-facing: it seeds the corrective retry prompt.
pub fn validate_answer(answer: &str, schema: &Value) -> Result<(), String> {
    let trimmed = answer.trim();
    let value: Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(e) => match strip_fence(trimmed).and_then(|inner| serde_json::from_str(inner).ok()) {
            Some(v) => v,
            None => return Err(format!("final answer is not valid JSON ({e})")),
        },
    };
    validate(&value, schema)
}

/// The JSON inside a ```-fenced block (any info string), if the whole answer is one.
fn strip_fence(t: &str) -> Option<&str> {
    let body = t.strip_prefix("```")?.strip_suffix("```")?;
    let nl = body.find('\n')?;
    Some(&body[nl + 1..])
}

pub fn validate(value: &Value, schema: &Value) -> Result<(), String> {
    check(value, schema, schema, "", 0)
}

/// Cyclic `$ref` backstop; also caps combinator recursion.
const MAX_DEPTH: usize = 64;

fn loc(path: &str) -> String {
    if path.is_empty() {
        "$".to_string()
    } else {
        format!("${path}")
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn matches_type(v: &Value, t: &str) -> bool {
    match t {
        "integer" => v.as_f64().is_some_and(|f| f.fract() == 0.0),
        "number" => v.is_number(),
        _ => type_name(v) == t,
    }
}

fn check(v: &Value, schema: &Value, root: &Value, path: &str, depth: usize) -> Result<(), String> {
    if depth > MAX_DEPTH {
        return Err(format!("{}: schema $ref/nesting too deep", loc(path)));
    }
    let s = match schema {
        Value::Bool(true) => return Ok(()),
        Value::Bool(false) => return Err(format!("{}: schema allows nothing here", loc(path))),
        Value::Object(m) => m,
        // Malformed schema node — permissive, like unknown keywords.
        _ => return Ok(()),
    };

    if let Some(r) = s.get("$ref").and_then(Value::as_str) {
        let target = r
            .strip_prefix('#')
            .and_then(|p| root.pointer(p))
            .ok_or_else(|| format!("{}: unresolvable $ref {r}", loc(path)))?;
        check(v, target, root, path, depth + 1)?;
    }

    if let Some(t) = s.get("type") {
        let ok = match t {
            Value::String(t) => matches_type(v, t),
            Value::Array(ts) => ts
                .iter()
                .filter_map(Value::as_str)
                .any(|t| matches_type(v, t)),
            _ => true,
        };
        if !ok {
            return Err(format!("{}: expected {t}, got {}", loc(path), type_name(v)));
        }
    }
    if let Some(e) = s.get("enum").and_then(Value::as_array)
        && !e.iter().any(|c| c == v)
    {
        return Err(format!("{}: not one of the enum values", loc(path)));
    }
    if let Some(c) = s.get("const")
        && c != v
    {
        return Err(format!("{}: does not equal the const value", loc(path)));
    }

    match v {
        Value::Object(obj) => {
            if let Some(req) = s.get("required").and_then(Value::as_array) {
                for r in req.iter().filter_map(Value::as_str) {
                    if !obj.contains_key(r) {
                        return Err(format!("{}: missing required property \"{r}\"", loc(path)));
                    }
                }
            }
            let props = s.get("properties").and_then(Value::as_object);
            if let Some(props) = props {
                for (k, sub) in props {
                    if let Some(pv) = obj.get(k) {
                        check(pv, sub, root, &format!("{path}/{k}"), depth + 1)?;
                    }
                }
            }
            match s.get("additionalProperties") {
                Some(Value::Bool(false)) => {
                    for k in obj.keys() {
                        if !props.is_some_and(|p| p.contains_key(k)) {
                            return Err(format!("{}: unexpected property \"{k}\"", loc(path)));
                        }
                    }
                }
                Some(ap @ Value::Object(_)) => {
                    for (k, pv) in obj {
                        if !props.is_some_and(|p| p.contains_key(k)) {
                            check(pv, ap, root, &format!("{path}/{k}"), depth + 1)?;
                        }
                    }
                }
                _ => {}
            }
        }
        Value::Array(items) => {
            if let Some(min) = s.get("minItems").and_then(Value::as_u64)
                && (items.len() as u64) < min
            {
                return Err(format!("{}: fewer than {min} items", loc(path)));
            }
            if let Some(max) = s.get("maxItems").and_then(Value::as_u64)
                && (items.len() as u64) > max
            {
                return Err(format!("{}: more than {max} items", loc(path)));
            }
            if let Some(item_schema) = s.get("items") {
                for (i, item) in items.iter().enumerate() {
                    check(item, item_schema, root, &format!("{path}/{i}"), depth + 1)?;
                }
            }
        }
        Value::String(text) => {
            let len = text.chars().count() as u64;
            if let Some(min) = s.get("minLength").and_then(Value::as_u64)
                && len < min
            {
                return Err(format!("{}: shorter than {min} characters", loc(path)));
            }
            if let Some(max) = s.get("maxLength").and_then(Value::as_u64)
                && len > max
            {
                return Err(format!("{}: longer than {max} characters", loc(path)));
            }
        }
        Value::Number(n) => {
            let f = n.as_f64().unwrap_or(0.0);
            if let Some(min) = s.get("minimum").and_then(Value::as_f64)
                && f < min
            {
                return Err(format!("{}: below minimum {min}", loc(path)));
            }
            if let Some(max) = s.get("maximum").and_then(Value::as_f64)
                && f > max
            {
                return Err(format!("{}: above maximum {max}", loc(path)));
            }
            if let Some(min) = s.get("exclusiveMinimum").and_then(Value::as_f64)
                && f <= min
            {
                return Err(format!("{}: not above exclusiveMinimum {min}", loc(path)));
            }
            if let Some(max) = s.get("exclusiveMaximum").and_then(Value::as_f64)
                && f >= max
            {
                return Err(format!("{}: not below exclusiveMaximum {max}", loc(path)));
            }
        }
        _ => {}
    }

    if let Some(all) = s.get("allOf").and_then(Value::as_array) {
        for sub in all {
            check(v, sub, root, path, depth + 1)?;
        }
    }
    if let Some(any) = s.get("anyOf").and_then(Value::as_array)
        && !any
            .iter()
            .any(|sub| check(v, sub, root, path, depth + 1).is_ok())
    {
        return Err(format!("{}: matches none of the anyOf schemas", loc(path)));
    }
    if let Some(one) = s.get("oneOf").and_then(Value::as_array) {
        let n = one
            .iter()
            .filter(|sub| check(v, sub, root, path, depth + 1).is_ok())
            .count();
        if n != 1 {
            return Err(format!(
                "{}: matches {n} of the oneOf schemas (need exactly 1)",
                loc(path)
            ));
        }
    }
    if let Some(not) = s.get("not")
        && check(v, not, root, path, depth + 1).is_ok()
    {
        return Err(format!("{}: matches the `not` schema", loc(path)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn types_and_required_properties() {
        let schema = json!({
            "type": "object",
            "required": ["name", "count"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "count": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": false,
        });
        assert!(validate(&json!({"name": "a", "count": 3}), &schema).is_ok());
        assert!(validate(&json!({"name": "a", "count": 3.0}), &schema).is_ok());
        let missing = validate(&json!({"name": "a"}), &schema).unwrap_err();
        assert!(missing.contains("count"), "{missing}");
        let wrong = validate(&json!({"name": "a", "count": "3"}), &schema).unwrap_err();
        assert!(wrong.contains("$/count"), "{wrong}");
        let extra = validate(&json!({"name": "a", "count": 1, "x": 1}), &schema).unwrap_err();
        assert!(extra.contains("unexpected property"), "{extra}");
        assert!(validate(&json!({"name": "", "count": 1}), &schema).is_err());
        assert!(validate(&json!({"name": "a", "count": -1}), &schema).is_err());
    }

    #[test]
    fn arrays_enums_and_bounds() {
        let schema = json!({
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "items": {"enum": ["a", "b"]},
        });
        assert!(validate(&json!(["a", "b"]), &schema).is_ok());
        assert!(validate(&json!([]), &schema).is_err());
        assert!(validate(&json!(["a", "b", "a"]), &schema).is_err());
        let bad = validate(&json!(["c"]), &schema).unwrap_err();
        assert!(bad.contains("$/0"), "{bad}");
    }

    #[test]
    fn combinators_and_local_ref() {
        let schema = json!({
            "$defs": {"id": {"type": "string", "minLength": 2}},
            "type": "object",
            "properties": {
                "id": {"$ref": "#/$defs/id"},
                "kind": {"oneOf": [{"const": "x"}, {"const": "y"}]},
                "tag": {"not": {"const": "forbidden"}},
            },
        });
        assert!(validate(&json!({"id": "ab", "kind": "x", "tag": "t"}), &schema).is_ok());
        assert!(validate(&json!({"id": "a"}), &schema).is_err());
        assert!(validate(&json!({"kind": "z"}), &schema).is_err());
        assert!(validate(&json!({"tag": "forbidden"}), &schema).is_err());
        let dangling = json!({"$ref": "#/$defs/nope"});
        assert!(validate(&json!(1), &dangling).is_err());
    }

    #[test]
    fn type_arrays_and_null() {
        let schema = json!({"type": ["string", "null"]});
        assert!(validate(&json!("s"), &schema).is_ok());
        assert!(validate(&json!(null), &schema).is_ok());
        assert!(validate(&json!(1), &schema).is_err());
    }

    #[test]
    fn unknown_keywords_are_ignored() {
        let schema = json!({"type": "string", "pattern": "[", "format": "email"});
        assert!(validate(&json!("anything"), &schema).is_ok());
    }

    #[test]
    fn answer_extraction_accepts_bare_and_fenced_json() {
        let schema = json!({"type": "object", "required": ["ok"]});
        assert!(validate_answer(r#"{"ok": true}"#, &schema).is_ok());
        assert!(validate_answer("```json\n{\"ok\": true}\n```", &schema).is_ok());
        assert!(validate_answer("```\n{\"ok\": true}\n```", &schema).is_ok());
        let prose = validate_answer("Here you go: {\"ok\": true}", &schema).unwrap_err();
        assert!(prose.contains("not valid JSON"), "{prose}");
    }

    #[test]
    fn cyclic_ref_terminates() {
        let schema = json!({"$defs": {"a": {"$ref": "#/$defs/a"}}, "$ref": "#/$defs/a"});
        let err = validate(&json!(1), &schema).unwrap_err();
        assert!(err.contains("too deep"), "{err}");
    }
}
