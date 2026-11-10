//! Bounded global text index. Ordering never changes request contents or budgets.
use super::*;

struct Item {
    prefix: Vec<u32>,
    tokens: usize,
    output_limit: usize,
}
impl Item {
    fn work(&self) -> usize {
        // A cold prior, not the requested maximum as a length prediction.
        self.tokens.min(self.output_limit)
    }
}

/// A compressed prefix-tree traversal keeps each subtree together. Heavy
/// subtrees run first to avoid leaving the longest translations at the tail.
fn order(items: &[Item], share_prefix: bool) -> (Vec<usize>, Vec<usize>) {
    let mut lex: Vec<_> = (0..items.len()).collect();
    let rank = |&a: &usize, &b: &usize| {
        items[b]
            .work()
            .cmp(&items[a].work())
            .then_with(|| a.cmp(&b))
    };
    let mut hints = vec![0; items.len()];
    if !share_prefix {
        lex.sort_unstable_by(rank);
        return (lex, hints);
    }
    lex.sort_unstable_by(|&a, &b| items[a].prefix.cmp(&items[b].prefix).then(a.cmp(&b)));
    for pair in lex.windows(2) {
        let shared = items[pair[0]]
            .prefix
            .iter()
            .zip(&items[pair[1]].prefix)
            .take_while(|(a, b)| a == b)
            .count();
        if shared >= 32 {
            for &i in pair {
                hints[i] = hints[i].max(shared);
            }
        }
    }
    let mut pending = vec![(0, lex.len())];
    let mut result = Vec::with_capacity(lex.len());
    while let Some((start, end)) = pending.pop() {
        if start == end {
            continue;
        }
        let first = &items[lex[start]].prefix;
        let last = &items[lex[end - 1]].prefix;
        let depth = first.iter().zip(last).take_while(|(a, b)| a == b).count();
        if first == last {
            lex[start..end].sort_unstable_by(rank);
            result.extend_from_slice(&lex[start..end]);
            continue;
        }
        let mut groups = vec![];
        let mut left = start;
        while left < end {
            let key = items[lex[left]].prefix.get(depth);
            let mut right = left + 1;
            let mut work = items[lex[left]].work();
            while right < end && items[lex[right]].prefix.get(depth) == key {
                work = work.max(items[lex[right]].work());
                right += 1;
            }
            groups.push((work, left, right));
            left = right;
        }
        // Stack visits the largest subtree first, with stable ties.
        groups.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
        pending.extend(groups.into_iter().map(|(_, start, end)| (start, end)));
    }
    (result, hints)
}

pub(super) async fn prepare(
    input: &mut Input,
    state: &Service,
    defaults: &serde_json::Map<String, Value>,
    share_prefix: bool,
    report: &mut BatchReport,
) -> Result<()> {
    let started = Instant::now();
    let count = input.ids.len();
    // Bound tokens and tree/index overhead to one quarter of the configured
    // preprocessing budget. Huge files keep the streaming path instead.
    let budget = state.limits.memory_mib * (1 << 20) / 4;
    let Some(bytes) = budget.checked_sub(count.saturating_mul(256)) else {
        return Ok(());
    };
    let depth = (bytes / count.max(1) / 4).min(4096);
    if depth == 0 {
        return Ok(());
    }
    let mut items = Vec::with_capacity(count);
    while let Some(record) = input.next()? {
        let prepared = (|| {
            let request = request(record.body?, &state.model, defaults)?;
            // Planning never fetches media twice or retains image tensors.
            if request.messages.iter().any(|m| !m["content"].is_string()) {
                return Err("Media/structured content uses bounded streaming preparation".into());
            }
            if preparation::request_memory_mib(
                &request,
                state.context,
                state.codec.max_tokens(&request),
            ) as usize
                > state.limits.memory_mib
            {
                return Err("Request exceeds preprocessing memory budget".into());
            }
            state.codec.prepare(
                request,
                &state.model,
                state.context,
                state.vision.as_ref(),
                &mut preparation::Context::new(
                    Arc::new(AtomicBool::new(false)),
                    state.limits.memory_mib * (1 << 20),
                ),
            )
        })();
        items.push(match prepared {
            Ok(prepared) => {
                report.planned_requests += 1;
                let mut prefix = prepared.input;
                let tokens = prefix.len();
                prefix.truncate(depth);
                prefix.shrink_to_fit();
                report.planning_prefix_tokens += prefix.len();
                Item {
                    prefix,
                    tokens,
                    output_limit: prepared.max_tokens,
                }
            }
            Err(_) => Item {
                prefix: vec![],
                tokens: 0,
                output_limit: 0,
            },
        });
        // Let cancellation run between records, even if individual templates
        // or tokenization are expensive.
        tokio::task::yield_now().await;
    }
    let (order, hints) = order(&items, share_prefix);
    input.reorder(
        order,
        hints
            .into_iter()
            .enumerate()
            .map(|(i, hint)| (!items[i].prefix.is_empty()).then_some(hint))
            .collect(),
    )?;
    report.planning_s = started.elapsed().as_secs_f64();
    eprintln!(
        "BATCH PLAN {} text requests, {} retained prefix tokens, {:.3}s",
        report.planned_requests, report.planning_prefix_tokens, report.planning_s
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(prefix: &[u32], tokens: usize) -> Item {
        Item {
            prefix: prefix.to_vec(),
            tokens,
            output_limit: 4096,
        }
    }
    #[test]
    fn heavy_prefix_subtrees_stay_contiguous_and_longest_leaves_go_first() {
        let items = vec![
            item(&[1, 2, 3], 40),
            item(&[9, 2], 90),
            item(&[1, 2, 4], 100),
            item(&[9, 3], 80),
            item(&[1, 2, 3], 60),
        ];
        assert_eq!(order(&items, true).0, vec![2, 4, 0, 1, 3]);
        assert_eq!(order(&items, false).0, vec![2, 1, 3, 4, 0]);
        assert!(order(&[], true).0.is_empty());
    }
    #[test]
    fn hints_are_actual_token_prefixes_and_limits_are_not_length_predictions() {
        let mut common = vec![5; 65];
        let mut a = common.clone();
        a.push(2);
        let mut b = common.clone();
        b.push(3);
        common[63] = 7;
        let mut items = vec![item(&a, 100), item(&b, 100), item(&common, 100)];
        items[0].output_limit = 1024;
        items[1].output_limit = 4096;
        assert_eq!(order(&items, true).1, vec![65, 65, 63]);
        assert_eq!(order(&items, false).0, vec![0, 1, 2]);
    }
}
