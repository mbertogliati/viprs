mod args;
mod helpers;
mod pipeline;
mod runner;
mod types;

use std::path::Path;

use crate::common;

pub fn run(args: &[String]) {
    if let Some(message) = helpers::debug_build_error() {
        eprintln!("{message}");
        std::process::exit(1);
    }

    let bench_args = args::parse_args(args);
    let structured_output = bench_args.json_output || bench_args.ai_output;
    if bench_args.profile_only {
        if bench_args.multi_sizes
            || bench_args.scenario_set.is_some()
            || bench_args.parameter_matrix
            || bench_args.json_output
            || bench_args.check_regression
            || bench_args.profile_stages
            || bench_args.ai_output
        {
            eprintln!(
                "--profile-only only supports a single viprs scenario with no comparison/reporting flags"
            );
            std::process::exit(1);
        }

        let input_path = common::resolve_input(&bench_args.input);
        if !input_path.exists() {
            eprintln!("Input file not found: {}", input_path.display());
            std::process::exit(1);
        }

        runner::run_viprs_profile_only(
            &input_path,
            &bench_args.op,
            &bench_args.op_args,
            bench_args.iterations,
            bench_args.threads,
        );
        return;
    }

    if bench_args.parameter_matrix && !bench_args.op_args.is_empty() {
        eprintln!(
            "--matrix expands the canonical scenario set for '{}' and does not accept explicit op args",
            bench_args.op
        );
        std::process::exit(1);
    }
    let mut summary_rows = Vec::new();
    let scenario_arg_sets = if bench_args.parameter_matrix {
        match helpers::canonical_op_arg_matrix(&bench_args.op) {
            Some(matrix) => matrix,
            None => {
                eprintln!(
                    "--matrix is not supported for '{}'. Supported ops: thumbnail, gauss_blur, colourspace, dilate, erode, median_blur",
                    bench_args.op
                );
                std::process::exit(1);
            }
        }
    } else {
        vec![bench_args.op_args.clone()]
    };

    if let Some(scenario_set) = bench_args.scenario_set.as_deref() {
        for op_args in &scenario_arg_sets {
            summary_rows.extend(runner::run_scenario_set(
                scenario_set,
                &bench_args.op,
                op_args,
                bench_args.iterations,
                bench_args.threads,
                bench_args.e2e,
                structured_output,
            ));
        }
    } else if bench_args.multi_sizes {
        for op_args in &scenario_arg_sets {
            summary_rows.extend(runner::run_multi_size(
                &bench_args.op,
                op_args,
                bench_args.iterations,
                bench_args.threads,
                bench_args.e2e,
                structured_output,
            ));
        }
    } else {
        let input_path = common::resolve_input(&bench_args.input);
        let effective_threads =
            helpers::default_bench_threads(bench_args.threads, &bench_args.op, &input_path);

        if !input_path.exists() {
            eprintln!("Input file not found: {}", input_path.display());
            std::process::exit(1);
        }

        let results_dir = common::repo_root().join("tools/bench-vs-libvips/results");
        std::fs::create_dir_all(&results_dir).ok();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for op_args in &scenario_arg_sets {
            if !structured_output {
                println!(
                    "=== xtask bench {} ===",
                    helpers::scenario_display_label(&bench_args.op, op_args)
                );
            }
            let cmp = runner::run_single(
                &input_path,
                &bench_args.op,
                op_args,
                bench_args.iterations,
                effective_threads,
                bench_args.e2e,
                structured_output,
            );

            if bench_args.profile_stages {
                runner::print_thumbnail_stage_profile(
                    &input_path,
                    op_args,
                    bench_args.iterations,
                    effective_threads,
                    bench_args.ai_output,
                );
            }

            let scenario_slug = helpers::scenario_slug(&bench_args.op, op_args);
            let result_file = results_dir.join(format!("{scenario_slug}_{timestamp}.json"));
            let json = serde_json::to_string_pretty(&cmp).unwrap_or_default();
            std::fs::write(&result_file, &json).expect("Failed to write results");
            if !structured_output {
                println!("Results saved: {}", result_file.display());
                println!();
            }
            summary_rows.push(helpers::build_summary_row(
                &bench_args.op,
                op_args,
                &input_path,
                None,
                helpers::infer_square_size_from_input(&input_path),
                &cmp,
            ));
        }
    }

    if bench_args.json_output {
        let json_output = if summary_rows.len() == 1 {
            serde_json::to_string_pretty(&summary_rows[0])
        } else {
            serde_json::to_string_pretty(&summary_rows)
        };
        match json_output {
            Ok(json) => println!("{json}"),
            Err(error) => {
                eprintln!("Failed to serialize summary JSON: {error}");
                std::process::exit(1);
            }
        }
    }

    if bench_args.check_regression {
        if summary_rows.is_empty() {
            eprintln!("No benchmark rows were produced; regression check cannot run.");
            std::process::exit(1);
        }
        let baseline_path = bench_args
            .baseline
            .as_deref()
            .expect("validated by parse_args");
        let baseline_rows = match helpers::load_baseline_rows(baseline_path) {
            Ok(rows) => rows,
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        };
        let failures = helpers::regression_messages(&summary_rows, &baseline_rows);
        if failures.is_empty() {
            if !structured_output {
                println!("Regression check passed.");
            }
        } else {
            for failure in &failures {
                eprintln!("REGRESSION: {failure}");
            }
            std::process::exit(1);
        }
    }
}

pub fn run_viprs_alloc_only(input_path: &Path, op: &str, op_args: &[String], iterations: usize) {
    let requested_threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let effective_threads = helpers::default_bench_threads(requested_threads, op, input_path);
    runner::run_viprs_profile_only(input_path, op, op_args, iterations, effective_threads);
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::os::unix::process::ExitStatusExt;
    use std::process::{ExitStatus, Output};

    use super::args::parse_args;
    use super::helpers::{
        DEBUG_BENCH_ERROR, DEFAULT_ITERATIONS, canonical_op_arg_matrix, debug_build_error,
        parse_summary_rows, regression_messages, scenario_set_supports_op, summary_key,
    };
    use super::pipeline::{build_viprs_composite_pipeline, build_viprs_pipeline};
    use super::runner::{BaselineBackend, parse_baseline_output};
    use super::types::SummaryRow;
    use crate::common::resolve_input;

    #[test]
    fn debug_build_guard_matches_current_profile() {
        if cfg!(debug_assertions) {
            assert_eq!(debug_build_error(), Some(DEBUG_BENCH_ERROR));
        } else {
            assert_eq!(debug_build_error(), None);
        }
    }

    #[test]
    fn parse_args_defaults_are_e2e() {
        let args = vec!["input.jpg".to_owned(), "invert".to_owned()];
        let parsed = parse_args(&args);
        assert_eq!(parsed.iterations, DEFAULT_ITERATIONS);
        assert!(parsed.threads > 0);
        assert!(parsed.e2e, "e2e should be on by default");
    }

    #[test]
    fn parse_args_no_e2e_opt_out() {
        let args = vec![
            "input.jpg".to_owned(),
            "invert".to_owned(),
            "--no-e2e".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert!(!parsed.e2e);
    }

    #[test]
    fn parse_args_profile_only_implies_no_e2e() {
        let args = vec![
            "input.jpg".to_owned(),
            "invert".to_owned(),
            "--profile-only".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert!(parsed.profile_only);
        assert!(!parsed.e2e);
    }

    #[test]
    fn parse_args_normalizes_extract_area_alias() {
        let args = vec!["input.jpg".to_owned(), "extract_area".to_owned()];
        let parsed = parse_args(&args);
        assert_eq!(parsed.op, "extract-area");
    }

    #[test]
    fn parse_args_normalizes_load_exr_alias() {
        let args = vec!["input.exr".to_owned(), "load_exr".to_owned()];
        let parsed = parse_args(&args);
        assert_eq!(parsed.op, "load-exr");
    }

    #[test]
    fn parse_args_thumbnail_stage_profile_requires_thumbnail() {
        let args = vec![
            "input.jpg".to_owned(),
            "thumbnail".to_owned(),
            "400".to_owned(),
            "--no-e2e".to_owned(),
            "--profile-stages".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert_eq!(parsed.op, "thumbnail");
        assert!(parsed.profile_stages);
        assert!(!parsed.e2e);
    }

    #[test]
    fn parse_args_json_and_regression_flags() {
        let args = vec![
            "input.jpg".to_owned(),
            "invert".to_owned(),
            "--sizes".to_owned(),
            "--json".to_owned(),
            "--check-regression".to_owned(),
            "--baseline".to_owned(),
            "baseline.json".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert!(parsed.multi_sizes);
        assert!(parsed.json_output);
        assert!(parsed.check_regression);
        assert_eq!(
            parsed.baseline.as_deref().and_then(|path| path.to_str()),
            Some("baseline.json")
        );
    }

    #[test]
    fn parse_args_scenario_set() {
        let args = vec![
            "ignored".to_owned(),
            "invert".to_owned(),
            "--scenario-set".to_owned(),
            "compute-baselines".to_owned(),
            "--iterations".to_owned(),
            "3".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert_eq!(parsed.scenario_set.as_deref(), Some("compute-baselines"));
        assert!(!parsed.multi_sizes);
        assert_eq!(parsed.iterations, 3);
    }

    #[test]
    fn parse_args_matrix_flag() {
        let args = vec![
            "input.jpg".to_owned(),
            "thumbnail".to_owned(),
            "--matrix".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert!(parsed.parameter_matrix);
        assert!(parsed.op_args.is_empty());
    }

    #[test]
    fn parse_args_save_avif_opt_outs() {
        let args = vec![
            "input.avif".to_owned(),
            "save-avif".to_owned(),
            "--no-e2e".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert_eq!(parsed.op, "save-avif");
        assert!(parsed.op_args.is_empty());
        assert!(!parsed.e2e);
    }

    #[test]
    fn parse_args_save_avif_u16_arg() {
        let args = vec![
            "input.png".to_owned(),
            "save-avif".to_owned(),
            "u16".to_owned(),
            "--no-e2e".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert_eq!(parsed.op, "save-avif");
        assert_eq!(parsed.op_args, vec!["u16".to_owned()]);
        assert!(!parsed.e2e);
    }

    #[test]
    fn parse_args_save_gif_opt_outs() {
        let args = vec![
            "input.gif".to_owned(),
            "save-gif".to_owned(),
            "--no-e2e".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert_eq!(parsed.op, "save-gif");
        assert!(parsed.op_args.is_empty());
        assert!(!parsed.e2e);
    }

    #[test]
    fn parse_args_save_heif_opt_outs() {
        let args = vec![
            "input.jpg".to_owned(),
            "save-heif".to_owned(),
            "--no-e2e".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert_eq!(parsed.op, "save-heif");
        assert!(parsed.op_args.is_empty());
        assert!(!parsed.e2e);
    }

    #[test]
    fn parse_args_save_exr_opt_outs() {
        let args = vec![
            "input.exr".to_owned(),
            "save-exr".to_owned(),
            "--no-e2e".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert_eq!(parsed.op, "save-exr");
        assert!(parsed.op_args.is_empty());
        assert!(!parsed.e2e);
    }

    #[test]
    fn save_exr_uses_openexr_baseline_backend() {
        assert_eq!(
            BaselineBackend::for_op("save-exr"),
            BaselineBackend::OpenExr
        );
        assert_eq!(BaselineBackend::for_op("invert"), BaselineBackend::Libvips);
    }

    #[test]
    fn baseline_runner_crash_is_reported_as_error() {
        let output = Output {
            status: ExitStatus::from_raw(256),
            stdout: Vec::new(),
            stderr: b"Operation failed at iteration 0\n".to_vec(),
        };

        let error = match parse_baseline_output(BaselineBackend::OpenExr, &output) {
            Ok(_) => panic!("runner failures must bubble up"),
            Err(error) => error,
        };
        assert!(error.contains("openexr-runner failed"));
        assert!(error.contains("iteration 0"));
    }

    #[test]
    fn parse_args_save_jp2k_opt_outs() {
        let args = vec![
            "input.jpg".to_owned(),
            "save-jp2k".to_owned(),
            "--iterations".to_owned(),
            "7".to_owned(),
            "--no-e2e".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert_eq!(parsed.op, "save-jp2k");
        assert_eq!(parsed.iterations, 7);
        assert!(parsed.op_args.is_empty());
        assert!(!parsed.e2e);
    }

    #[test]
    fn parse_args_save_tiff_opt_outs() {
        let args = vec![
            "input.jpg".to_owned(),
            "save-tiff".to_owned(),
            "--no-e2e".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert_eq!(parsed.op, "save-tiff");
        assert!(parsed.op_args.is_empty());
        assert!(!parsed.e2e);
    }

    #[test]
    fn parse_args_save_tiff_compression_arg() {
        let args = vec![
            "input.jpg".to_owned(),
            "save-tiff".to_owned(),
            "jpeg".to_owned(),
            "--no-e2e".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert_eq!(parsed.op, "save-tiff");
        assert_eq!(parsed.op_args, vec!["jpeg".to_owned()]);
        assert!(!parsed.e2e);
    }

    #[test]
    fn parse_args_shrink_accepts_optional_vertical_factor() {
        let args = vec![
            "input.jpg".to_owned(),
            "shrink".to_owned(),
            "5".to_owned(),
            "3".to_owned(),
            "--no-e2e".to_owned(),
        ];
        let parsed = parse_args(&args);
        assert_eq!(parsed.op, "shrink");
        assert_eq!(parsed.op_args, vec!["5".to_owned(), "3".to_owned()]);
        assert!(!parsed.e2e);
    }

    fn scenario_set_supports_composite_shrink() {
        assert!(scenario_set_supports_op("shrink"));
    }

    #[test]
    fn scenario_set_supports_colourspace() {
        assert!(scenario_set_supports_op("colourspace"));
    }

    #[test]
    fn scenario_set_supports_float_arithmetic_ops() {
        assert!(scenario_set_supports_op("abs"));
        assert!(scenario_set_supports_op("add"));
        assert!(scenario_set_supports_op("multiply"));
        assert!(scenario_set_supports_op("round"));
        assert!(scenario_set_supports_op("floor"));
    }

    #[test]
    fn scenario_set_supports_zoom() {
        assert!(scenario_set_supports_op("zoom"));
    }

    #[test]
    fn parse_summary_rows_accepts_single_object() {
        let json = r#"{
            "op": "invert",
            "input": "tests/fixtures/images/bench_512x512.jpg",
            "size": 512,
            "viprs_p50_ms": 1.23,
            "viprs_p95_ms": 1.31,
            "libvips_p50_ms": 1.45,
            "libvips_p95_ms": 1.54,
            "ratio": 0.848,
            "ratio_p95": 0.851
        }"#;

        let rows = parse_summary_rows(json).expect("single object should parse");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].size, Some(512));
    }

    #[test]
    fn regression_messages_detects_ratio_regressions() {
        let baseline = vec![SummaryRow {
            op: "invert".to_owned(),
            op_args: Vec::new(),
            input: "tests/fixtures/images/bench_512x512.jpg".to_owned(),
            scenario: None,
            size: Some(512),
            viprs_p50_ms: 1.2,
            viprs_p95_ms: 1.3,
            libvips_p50_ms: Some(1.5),
            libvips_p95_ms: Some(1.6),
            ratio: Some(0.80),
            ratio_p95: Some(0.81),
        }];
        let current = vec![SummaryRow {
            ratio: Some(0.90),
            ..baseline[0].clone()
        }];

        let failures = regression_messages(&current, &baseline);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("regressed"));
    }

    #[test]
    fn regression_messages_allows_small_ratio_changes() {
        let baseline = vec![SummaryRow {
            op: "invert".to_owned(),
            op_args: Vec::new(),
            input: "tests/fixtures/images/bench_512x512.jpg".to_owned(),
            scenario: None,
            size: Some(512),
            viprs_p50_ms: 1.2,
            viprs_p95_ms: 1.3,
            libvips_p50_ms: Some(1.5),
            libvips_p95_ms: Some(1.6),
            ratio: Some(0.80),
            ratio_p95: Some(0.81),
        }];
        let current = vec![SummaryRow {
            ratio: Some(0.88),
            ..baseline[0].clone()
        }];

        assert!(regression_messages(&current, &baseline).is_empty());
    }

    #[test]
    fn build_viprs_pipeline_creates_internal_tile_cache() {
        let input = resolve_input("tests/fixtures/images/bench_512x512_rgba.png");
        let pipeline =
            build_viprs_composite_pipeline(&input, viprs::domain::ops::conversion::BlendMode::Over);

        assert!(
            pipeline.tile_cache.is_some(),
            "pipeline cache should be enabled"
        );
    }

    #[test]
    fn canonical_op_arg_matrix_covers_thumbnail_targets() {
        let matrix = canonical_op_arg_matrix("thumbnail").expect("thumbnail matrix");
        assert_eq!(
            matrix,
            vec![
                vec!["100".to_owned()],
                vec!["200".to_owned()],
                vec!["400".to_owned()],
                vec!["800".to_owned()],
            ]
        );
    }

    #[test]
    fn canonical_op_arg_matrix_covers_gauss_blur_sigmas() {
        let matrix = canonical_op_arg_matrix("gauss_blur").expect("gauss_blur matrix");
        assert_eq!(
            matrix,
            vec![
                vec!["0.5".to_owned()],
                vec!["1.0".to_owned()],
                vec!["2.0".to_owned()],
                vec!["5.0".to_owned()],
            ]
        );
    }

    #[test]
    fn canonical_op_arg_matrix_covers_colourspace_routes() {
        let matrix = canonical_op_arg_matrix("colourspace").expect("colourspace matrix");
        assert_eq!(
            matrix,
            vec![
                vec!["lab".to_owned()],
                vec!["lab".to_owned(), "srgb".to_owned()],
                vec!["xyz".to_owned()],
                vec!["xyz".to_owned(), "srgb".to_owned()],
                vec!["cmyk".to_owned()],
                vec!["cmyk".to_owned(), "srgb".to_owned()],
                vec!["hsv".to_owned()],
                vec!["hsv".to_owned(), "srgb".to_owned()],
                vec!["scrgb".to_owned()],
                vec!["scrgb".to_owned(), "srgb".to_owned()],
            ]
        );
    }

    #[test]
    fn canonical_op_arg_matrix_covers_morphology_kernel_sizes() {
        let expected = vec![vec!["3".to_owned()], vec!["5".to_owned()]];

        assert_eq!(
            canonical_op_arg_matrix("dilate").expect("dilate matrix"),
            expected
        );
        assert_eq!(
            canonical_op_arg_matrix("erode").expect("erode matrix"),
            vec![vec!["3".to_owned()], vec!["5".to_owned()]]
        );
        assert_eq!(
            canonical_op_arg_matrix("median_blur").expect("median_blur matrix"),
            expected
        );
    }

    #[test]
    fn summary_key_distinguishes_parameterized_scenarios() {
        let thumbnail_100 = SummaryRow {
            op: "thumbnail".to_owned(),
            op_args: vec!["100".to_owned()],
            input: "tests/fixtures/images/bench_512x512.jpg".to_owned(),
            scenario: None,
            size: Some(512),
            viprs_p50_ms: 1.0,
            viprs_p95_ms: 1.0,
            libvips_p50_ms: Some(1.0),
            libvips_p95_ms: Some(1.0),
            ratio: Some(1.0),
            ratio_p95: Some(1.0),
        };
        let thumbnail_400 = SummaryRow {
            op_args: vec!["400".to_owned()],
            ..thumbnail_100.clone()
        };

        assert_ne!(summary_key(&thumbnail_100), summary_key(&thumbnail_400));
    }

    #[test]
    fn build_viprs_pipeline_accepts_composite_shrink() {
        let input = resolve_input("tests/fixtures/images/bench_512x512.jpg");
        let op_args = vec!["3".to_owned(), "2".to_owned()];

        let pipeline = build_viprs_pipeline(&input, "shrink", &op_args);

        assert_eq!(pipeline.nodes.len(), 1);
    }

    #[test]
    fn build_viprs_pipeline_accepts_zoom() {
        let input = resolve_input("tests/fixtures/images/bench_512x512.jpg");
        let op_args = vec!["2".to_owned(), "2".to_owned()];

        let pipeline = build_viprs_pipeline(&input, "zoom", &op_args);

        assert_eq!(pipeline.nodes.len(), 1);
    }

    #[test]
    fn build_viprs_pipeline_accepts_conv_sharpen3() {
        let input = resolve_input("tests/fixtures/images/bench_512x512.jpg");

        let pipeline = build_viprs_pipeline(&input, "conv_sharpen3", &[]);

        assert_eq!(pipeline.nodes.len(), 1);
    }

    #[test]
    fn build_viprs_pipeline_accepts_conv_sobel3() {
        let input = resolve_input("tests/fixtures/images/bench_512x512.jpg");

        let pipeline = build_viprs_pipeline(&input, "conv_sobel3", &[]);

        assert_eq!(pipeline.nodes.len(), 1);
    }

    #[test]
    fn build_viprs_pipeline_accepts_convolution_coverage_ops() {
        let input = resolve_input("tests/fixtures/images/bench_512x512.jpg");

        for (op, op_args) in [
            ("convolve", Vec::<String>::new()),
            ("sobel", Vec::<String>::new()),
            ("prewitt", Vec::<String>::new()),
            ("laplacian", Vec::<String>::new()),
            ("median_blur", vec!["3".to_owned()]),
            ("unsharp_mask", vec!["0.5".to_owned(), "3.0".to_owned()]),
        ] {
            let pipeline = build_viprs_pipeline(&input, op, &op_args);
            assert_eq!(pipeline.nodes.len(), 1, "{op} should compile to one node");
        }
    }

    #[test]
    fn build_viprs_pipeline_accepts_float_arithmetic_ops() {
        let input = resolve_input("tests/fixtures/images/bench_512x512.exr");

        for op in ["abs", "sign", "round", "floor", "ceil"] {
            let pipeline = build_viprs_pipeline(&input, op, &[]);
            assert_eq!(pipeline.nodes.len(), 1, "{op} should compile to one node");
        }
    }

    #[test]
    fn build_viprs_pipeline_accepts_composite() {
        let input = resolve_input("tests/fixtures/images/bench_512x512_rgba.png");

        let pipeline =
            build_viprs_composite_pipeline(&input, viprs::domain::ops::conversion::BlendMode::Over);

        assert!(pipeline.nodes.len() >= 3);
        assert_eq!(
            pipeline.output_format,
            viprs::domain::format::BandFormatId::F32
        );
    }

    #[test]
    fn build_viprs_pipeline_accepts_multi_operation_bench_scenarios() {
        let input = resolve_input("tests/fixtures/images/bench_512x512.jpg");

        for op in [
            "thumbnail_sharpen",
            "thumbnail_gauss_blur",
            "thumbnail_linear",
            "resize_colourspace",
            "embed",
            "extract-area",
            "embed_extract",
            "three_op_chain",
        ] {
            let pipeline = build_viprs_pipeline(&input, op, &[]);
            match op {
                "extract-area" => {
                    assert_eq!(pipeline.width, 448, "extract-area should crop width");
                    assert_eq!(pipeline.height, 464, "extract-area should crop height");
                }
                "embed" => {
                    assert_eq!(pipeline.width, 768, "embed should expand width");
                    assert_eq!(pipeline.height, 704, "embed should expand height");
                    assert!(
                        pipeline.nodes.len() >= 1,
                        "{op} should compile to at least one pipeline node"
                    );
                }
                _ => assert!(
                    pipeline.nodes.len() >= 2,
                    "{op} should compile to a chained benchmark pipeline"
                ),
            }
        }
    }
}
