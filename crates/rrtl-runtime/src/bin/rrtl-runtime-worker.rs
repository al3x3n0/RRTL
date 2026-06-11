use std::{
    env,
    io::{self, Write},
    net::SocketAddr,
    time::Duration,
};

use rrtl_runtime::{TcpRuntimeTransportConfig, TcpRuntimeWorkerServer};

#[derive(Clone, Debug, PartialEq, Eq)]
struct WorkerConfig {
    transport: TcpRuntimeTransportConfig,
    max_connections: Option<usize>,
}

fn main() {
    if let Err(err) = run(env::args().skip(1), &mut io::stdout()) {
        eprintln!("rrtl-runtime-worker: {err}");
        std::process::exit(1);
    }
}

fn run(
    args: impl IntoIterator<Item = String>,
    stdout: &mut impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let args = args.into_iter().collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        writeln!(stdout, "{}", usage())?;
        return Ok(());
    }

    let config = WorkerConfig::parse(args)?;
    let mut server = TcpRuntimeWorkerServer::bind_with_config(config.transport)?;
    writeln!(stdout, "{{\"addr\":\"{}\"}}", server.local_addr()?)?;
    stdout.flush()?;

    match config.max_connections {
        Some(connections) => server.serve_connections(connections)?,
        None => server.serve()?,
    }
    Ok(())
}

impl WorkerConfig {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut config = TcpRuntimeTransportConfig::default();
        let mut max_connections = None;
        let mut args = args.into_iter();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--bind" => {
                    config.bind_addr = next_value(&mut args, "--bind")?
                        .parse::<SocketAddr>()
                        .map_err(|err| format!("invalid --bind socket address: {err}"))?;
                }
                "--once" => {
                    set_max_connections(&mut max_connections, 1)?;
                }
                "--max-connections" => {
                    let value = next_value(&mut args, "--max-connections")?;
                    let parsed = value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --max-connections value: {err}"))?;
                    if parsed == 0 {
                        return Err("--max-connections must be greater than zero".to_string());
                    }
                    set_max_connections(&mut max_connections, parsed)?;
                }
                "--read-timeout-ms" => {
                    config.read_timeout =
                        parse_timeout(&next_value(&mut args, "--read-timeout-ms")?)?;
                }
                "--write-timeout-ms" => {
                    config.write_timeout =
                        parse_timeout(&next_value(&mut args, "--write-timeout-ms")?)?;
                }
                other => {
                    return Err(format!("unknown argument `{other}`\n{}", usage()));
                }
            }
        }

        Ok(Self {
            transport: config,
            max_connections,
        })
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value\n{}", usage()))
}

fn set_max_connections(slot: &mut Option<usize>, value: usize) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err("--once and --max-connections cannot be used more than once".to_string());
    }
    Ok(())
}

fn parse_timeout(value: &str) -> Result<Option<Duration>, String> {
    if value == "none" {
        return Ok(None);
    }
    let millis = value
        .parse::<u64>()
        .map_err(|err| format!("invalid timeout value `{value}`: {err}"))?;
    Ok(Some(Duration::from_millis(millis)))
}

fn usage() -> String {
    "usage: rrtl-runtime-worker [--bind <addr>] [--once|--max-connections <n>] [--read-timeout-ms <ms|none>] [--write-timeout-ms <ms|none>]".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn parses_defaults() {
        let config = WorkerConfig::parse(args(&[])).unwrap();
        assert_eq!(
            config.transport.bind_addr,
            SocketAddr::from(([127, 0, 0, 1], 0))
        );
        assert_eq!(config.max_connections, None);
        assert_eq!(config.transport.read_timeout, Some(Duration::from_secs(30)));
        assert_eq!(
            config.transport.write_timeout,
            Some(Duration::from_secs(30))
        );
    }

    #[test]
    fn parses_bind_and_once() {
        let config = WorkerConfig::parse(args(&["--bind", "127.0.0.1:9000", "--once"])).unwrap();
        assert_eq!(
            config.transport.bind_addr,
            "127.0.0.1:9000".parse::<SocketAddr>().unwrap()
        );
        assert_eq!(config.max_connections, Some(1));
    }

    #[test]
    fn parses_max_connections_and_timeouts() {
        let config = WorkerConfig::parse(args(&[
            "--max-connections",
            "3",
            "--read-timeout-ms",
            "none",
            "--write-timeout-ms",
            "250",
        ]))
        .unwrap();
        assert_eq!(config.max_connections, Some(3));
        assert_eq!(config.transport.read_timeout, None);
        assert_eq!(
            config.transport.write_timeout,
            Some(Duration::from_millis(250))
        );
    }

    #[test]
    fn rejects_invalid_max_connections() {
        let err = WorkerConfig::parse(args(&["--max-connections", "0"])).unwrap_err();
        assert!(err.contains("greater than zero"));
    }

    #[test]
    fn rejects_duplicate_connection_bounds() {
        let err = WorkerConfig::parse(args(&["--once", "--max-connections", "2"])).unwrap_err();
        assert!(err.contains("cannot be used more than once"));
    }

    #[test]
    fn rejects_invalid_timeout() {
        let err = WorkerConfig::parse(args(&["--read-timeout-ms", "soon"])).unwrap_err();
        assert!(err.contains("invalid timeout"));
    }

    #[test]
    fn rejects_missing_values() {
        let err = WorkerConfig::parse(args(&["--bind"])).unwrap_err();
        assert!(err.contains("--bind requires a value"));
    }
}
