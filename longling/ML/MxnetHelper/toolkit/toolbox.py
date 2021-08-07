# coding: utf-8
# create by tongshiwei on 2020-11-13


__all__ = ["get_default_toolbox"]


def get_default_toolbox(
        loss_function=None,
        evaluation_formatter_parameters=None,
        progress_monitor_parameters=None,
        validation_logger_mode="w",
        silent=False,
        configuration=None
):  # pragma: no cover
    """
    New in version 1.3.16

    todo: consider whether to keep it

    Notice
    ------
    The developer who modify this document should simultaneously modify the related function in glue
    """

    from longling import path_append
    from longling.lib.clock import Clock
    from longling.lib.utilog import config_logging
    from longling.ML.toolkit import EpochEvalFMT as Formatter
    from longling.ML.toolkit import MovingLoss, ConsoleProgressMonitor as ProgressMonitor

    cfg = configuration

    toolbox = {
        "monitor": dict(),
        "timer": None,
        "formatter": dict(),
    }

    loss_monitor = MovingLoss(loss_function) if loss_function else None

    timer = Clock()

    progress_monitor = ProgressMonitor(
        indexes={
            "Loss": [name for name in loss_function]
        } if loss_function else {},
        values={
            "Loss": loss_monitor.losses
        } if loss_monitor else {},
        silent=silent,
        **progress_monitor_parameters if progress_monitor_parameters is not None else {}
    )

    validation_logger = config_logging(
        filename=path_append(cfg.model_dir, "result.log") if hasattr(cfg, "model_dir") else None,
        logger="%s-validation" % cfg.model_name if hasattr(cfg, "model_name") else "model",
        mode=validation_logger_mode,
        log_format="%(message)s",
    )

    # set evaluation formatter
    evaluation_formatter_parameters = {} \
        if evaluation_formatter_parameters is None \
        else evaluation_formatter_parameters

    evaluation_formatter = Formatter(
        logger=validation_logger,
        dump_file=getattr(cfg, "validation_result_file", False),
        **evaluation_formatter_parameters
    )

    toolbox["monitor"]["loss"] = loss_monitor
    toolbox["monitor"]["progress"] = progress_monitor
    toolbox["timer"] = timer
    toolbox["formatter"]["evaluation"] = evaluation_formatter

    return toolbox
