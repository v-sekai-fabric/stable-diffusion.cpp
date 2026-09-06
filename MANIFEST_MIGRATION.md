# Manifest migration notice

This repo's `ggml`, `examples/server/frontend`, `thirdparty/libwebp`, and
`thirdparty/libwebm` dependencies used to be git submodules; they now live
as projects in the `v-sekai-fabric/weftspun-keypoint` manifest at the same
on-disk paths (`3-interactor/stable-diffusion.cpp/<path>`).

If you consume this repo through the weftspun manifest, `repo sync` from
the workspace root populates them at the SHAs the manifest pins.

Direct-clone consumers (outside the manifest) need to check them out
manually. Current pins (2026-09-05):

- `ggml` → `v-sekai-fabric/sd-ggml` @ `e20c3a14aa70ee84ca58499814206dd08d8026bc`
- `examples/server/frontend` → `v-sekai-fabric/sdcpp-webui` @ `c4bce3d6b3f236614cca21014f076083b7270ba8`
- `thirdparty/libwebp` → `v-sekai-fabric/sd-libwebp` @ `0c9546f7efc61eac7f79ae115c3f99c91c21c443`
- `thirdparty/libwebm` → `v-sekai-fabric/sd-libwebm` @ `5bf12267eea773a32fcf4949de52b0add158a8d5`

Doctrine: workspace CLAUDE.md blocklists git submodules as a second
dependency mechanism `repo status` cannot see.
