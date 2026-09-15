# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run isolated no-video-path partial hardlock inference with ERP blended decoding."""

from pathlib import Path
from typing import Annotated, Union

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_transfer2.config import (
    BlurConfig,
    DepthConfig,
    EdgeConfig,
    SegConfig,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)
from cosmos_transfer2.config_no_video_path_blended_decode import (
    InferenceArgumentsNoVideoPathBlendedDecode,
    InferenceOverridesNoVideoPathBlendedDecode,
)

ControlUnion = Annotated[
    Union[
        Annotated[EdgeConfig, tyro.conf.subcommand("edge")],
        Annotated[DepthConfig, tyro.conf.subcommand("depth")],
        Annotated[BlurConfig, tyro.conf.subcommand("vis")],
        Annotated[SegConfig, tyro.conf.subcommand("seg")],
    ],
    tyro.conf.ConsolidateSubcommandArgs,
]


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    setup: SetupArguments
    overrides: InferenceOverridesNoVideoPathBlendedDecode
    control: ControlUnion = EdgeConfig()


def main(args: Args) -> None:
    samples, batch_hint_keys = InferenceArgumentsNoVideoPathBlendedDecode.from_files(
        args.input_files,
        overrides=args.overrides,
    )
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_transfer2.inference_partial_hardlock_no_video_path_blended_decode import (
        Control2WorldInferencePartialHardlockNoVideoPathBlendedDecode,
    )

    inference = Control2WorldInferencePartialHardlockNoVideoPathBlendedDecode(
        args.setup,
        batch_hint_keys=batch_hint_keys,
    )
    inference.generate(samples, output_dir=args.setup.output_dir)


if __name__ == "__main__":
    init_environment()
    try:
        main(
            tyro.cli(
                Args,
                description=__doc__,
                console_outputs=is_rank0(),
                config=(tyro.conf.OmitArgPrefixes,),
            )
        )
    except Exception as error:
        handle_tyro_exception(error)
    cleanup_environment()
