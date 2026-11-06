//! A bounded shared-value representation for final execution plans.
use crate::artifact::Result;
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    encoding: String,
    nodes: Vec<Value>,
    root: usize,
}

pub(crate) fn decode<T: DeserializeOwned>(raw: &[u8]) -> Result<T> {
    if !raw.starts_with(b"{\"encoding\":\"shared-json-v1\",") {
        return serde_json::from_slice(raw).map_err(|e| e.to_string());
    }
    let document: Document = serde_json::from_slice(raw).map_err(|e| e.to_string())?;
    if document.encoding != "shared-json-v1"
        || document.nodes.is_empty()
        || document.nodes.len() > 2_000_000
        || document.root >= document.nodes.len()
    {
        return Err("Invalid shared JSON document".into());
    }
    let nodes = &document.nodes;
    let mut sizes = Vec::<usize>::with_capacity(nodes.len());
    let mut depths = Vec::<usize>::with_capacity(nodes.len());
    for (index, node) in nodes.iter().enumerate() {
        let mut refs = Vec::new();
        match node {
            Value::Array(children) => refs.extend(children.iter()),
            Value::Object(children) => refs.extend(children.values()),
            _ => {}
        }
        let mut size = match node {
            Value::Object(children) => children.keys().map(|key| key.len() * 6 + 4).sum::<usize>(),
            Value::Array(_) => 2,
            _ => serde_json::to_vec(node).map_err(|e| e.to_string())?.len(),
        };
        let mut depth = 1;
        for child in refs {
            let child = child
                .as_u64()
                .and_then(|i| usize::try_from(i).ok())
                .filter(|&i| i < index)
                .ok_or("Shared JSON references must precede their parents")?;
            size = size
                .checked_add(sizes[child] + 1)
                .ok_or("Shared JSON size overflow")?;
            depth = depth.max(depths[child] + 1);
        }
        if size > 128 * 1024 * 1024 || depth > 128 {
            return Err("Shared JSON expansion exceeds its bounds".into());
        }
        sizes.push(size);
        depths.push(depth);
    }
    fn expand(nodes: &[Value], index: usize) -> Value {
        match &nodes[index] {
            Value::Array(children) => Value::Array(
                children
                    .iter()
                    .map(|i| expand(nodes, i.as_u64().unwrap() as usize))
                    .collect(),
            ),
            Value::Object(children) => Value::Object(
                children
                    .iter()
                    .map(|(key, i)| (key.clone(), expand(nodes, i.as_u64().unwrap() as usize)))
                    .collect(),
            ),
            value => value.clone(),
        }
    }
    serde_json::from_value(expand(nodes, document.root)).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn plan_values_and_order_survive_shared_references() {
        let raw =
            br#"{"encoding":"shared-json-v1","nodes":["op",1,1.0,[0,1,2],{"a":3,"b":3}],"root":4}"#;
        let value: Value = decode(raw).unwrap();
        assert_eq!(value, json!({"a":["op",1,1.0],"b":["op",1,1.0]}));
        let legacy: Value = decode(br#"{"a":[1,2],"b":true}"#).unwrap();
        assert_eq!(legacy, json!({"a":[1,2],"b":true}));
    }
    #[test]
    fn invalid_graphs_and_exponential_expansion_are_rejected() {
        for nodes in [
            json!([[0]]),
            json!([{"a":1},0]),
            json!([{"a":-1}]),
            json!([{"a":0.0}]),
        ] {
            let raw = format!("{{\"encoding\":\"shared-json-v1\",\"nodes\":{nodes},\"root\":0}}");
            assert!(decode::<Value>(raw.as_bytes()).is_err());
        }
        let mut nodes = vec![json!(0)];
        for i in 0..35 {
            nodes.push(json!([i, i]));
        }
        let raw = format!(
            "{{\"encoding\":\"shared-json-v1\",\"nodes\":{},\"root\":35}}",
            json!(nodes)
        );
        assert!(decode::<Value>(raw.as_bytes()).is_err());
    }
}
