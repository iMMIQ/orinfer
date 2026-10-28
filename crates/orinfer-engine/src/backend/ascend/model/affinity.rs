//! Optional CPU placement for the model's owned NPU worker threads.
use crate::artifact::Result;

const ENV: &str = "ORINFER_A2_WORKER_CPUS";
const CPU_LIMIT: usize = 1024;

pub(super) fn from_env(ranks: usize) -> Result<Option<Vec<Vec<usize>>>> {
    let value = match std::env::var(ENV) {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(error) => return Err(format!("{ENV}: {error}")),
    };
    parse(&value, ranks, &current()?).map(Some)
}

fn parse(value: &str, ranks: usize, allowed: &[usize]) -> Result<Vec<Vec<usize>>> {
    let groups: Vec<_> = value.split(';').collect();
    if groups.len() != ranks {
        return Err(format!(
            "{ENV} requires {ranks} semicolon-separated CPU lists"
        ));
    }
    groups
        .into_iter()
        .enumerate()
        .map(|(rank, group)| {
            let mut cpus = vec![];
            for item in group.split(',') {
                let number = |s: &str| {
                    s.trim()
                        .parse::<usize>()
                        .map_err(|_| format!("{ENV}: invalid CPU in rank {rank}"))
                };
                let (first, last) = match item.trim().split_once('-') {
                    Some((first, last)) => (number(first)?, number(last)?),
                    None => {
                        let cpu = number(item)?;
                        (cpu, cpu)
                    }
                };
                if first > last || last >= CPU_LIMIT {
                    return Err(format!("{ENV}: invalid CPU range in rank {rank}"));
                }
                for cpu in first..=last {
                    if !allowed.contains(&cpu) {
                        return Err(format!("{ENV}: CPU {cpu} is outside the current affinity"));
                    }
                    cpus.push(cpu);
                }
            }
            cpus.sort_unstable();
            cpus.dedup();
            Ok(cpus)
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn current() -> Result<Vec<usize>> {
    // SAFETY: cpu_set_t is an integer bitset whose empty representation is zero.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: pid 0 selects this thread; set is a complete writable bitset.
    if unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&set), &mut set) } != 0 {
        return Err(format!(
            "Read worker CPU affinity: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok((0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| {
            // SAFETY: cpu is below CPU_SETSIZE and set was initialized above.
            unsafe { libc::CPU_ISSET(cpu, &set) }
        })
        .collect())
}

#[cfg(not(target_os = "linux"))]
fn current() -> Result<Vec<usize>> {
    Err("A2 worker CPU affinity requires Linux".into())
}

#[cfg(target_os = "linux")]
pub(super) fn apply(cpus: &[usize]) -> Result<()> {
    if cpus.is_empty() || cpus.iter().any(|&cpu| cpu >= libc::CPU_SETSIZE as usize) {
        return Err("Invalid worker CPU affinity".into());
    }
    // SAFETY: cpu_set_t is an integer bitset whose empty representation is zero.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &cpu in cpus {
        // SAFETY: Every index is below CPU_SETSIZE and set is initialized.
        unsafe { libc::CPU_SET(cpu, &mut set) };
    }
    // SAFETY: pid 0 selects this owned thread; the kernel copies the live bitset.
    if unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set) } != 0 {
        return Err(format!(
            "Bind NPU worker CPUs: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub(super) fn apply(_: &[usize]) -> Result<()> {
    Err("A2 worker CPU affinity requires Linux".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_maps_accept_lists_ranges_and_duplicates() {
        assert_eq!(
            parse("0,2-3,2;4-5", 2, &[0, 2, 3, 4, 5]).unwrap(),
            vec![vec![0, 2, 3], vec![4, 5]],
        );
    }

    #[test]
    fn cpu_maps_reject_invalid_or_unavailable_cpus() {
        for value in [
            "", "0", ";1", "0;", "2-1;4", "1024;4", "0;6", "0-;4", "-1;4", "0,,1;4",
        ] {
            assert!(parse(value, 2, &[0, 1, 2, 3, 4, 5]).is_err(), "{value}");
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn binding_an_owned_thread_preserves_the_parent_affinity() {
        let parent = current().unwrap();
        let cpu = parent[0];
        std::thread::spawn(move || {
            apply(&[cpu]).unwrap();
            assert_eq!(current().unwrap(), vec![cpu]);
        })
        .join()
        .unwrap();
        assert_eq!(current().unwrap(), parent);
        assert!(apply(&[]).is_err());
        assert!(apply(&[libc::CPU_SETSIZE as usize]).is_err());
    }
}
