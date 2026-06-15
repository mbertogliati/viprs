mod args;
mod runner;
mod scenarios;

use crate::common;

pub fn run(args: &[String]) {
    let bench_args = args::parse_args(args);
    let input_path = common::resolve_input(&bench_args.input);

    if !input_path.exists() {
        eprintln!("Input file not found: {}", input_path.display());
        std::process::exit(1);
    }

    runner::run_web_bench(&input_path, &bench_args);
}
