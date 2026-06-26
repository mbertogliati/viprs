//! Compile-time public API guardrails for the first-class `ImagePipeline` surface.

#[test]
fn image_pipeline_public_api_contracts_compile_as_documented() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/ui/image_pipeline_encoded_input_api.rs");
    tests.pass("tests/ui/image_pipeline_load_api.rs");
    tests.pass("tests/ui/image_pipeline_dynamic_commit_api.rs");
    tests.pass("tests/ui/image_pipeline_fluent_api.rs");
    tests.pass("tests/ui/image_pipeline_raw_sink_api.rs");
    tests.pass("tests/ui/image_pipeline_intermediate_fusible_api.rs");
    tests.compile_fail("tests/ui/image_pipeline_requires_output_contract.rs");
    tests.compile_fail("tests/ui/pipeline_builder_root_not_public_dsl.rs");
    tests.compile_fail("tests/ui/pipeline_builder_not_public_dsl.rs");
}
