# import logging
# import Fcoloredlogs
# import sys
#
#
# def set_logging(level = logging.WARNING, name='project_logger'):
#     logging.basicConfig(level=logging.DEBUG, format=f"%(levelname)-8s: \t %(filename)s %(funcName)s %(lineno)s - %(message)s", datefmt="%H:%M:%S")
#     logger = logging.getLogger(name=name)
#
#     coloredlogs.install(logger=logger)
#     logger.propagate = False
#
#     coloredFormatter = coloredlogs.ColoredFormatter(
#     fmt='[%(name)s] %(levelname)-8s %(asctime)s %(funcName)s %(lineno)-3d  %(message)s',
#     level_styles=dict(
#         debug=dict(color='white'),
#         info=dict(color='blue'),
#         warning=dict(color='yellow', bright=True),
#         error=dict(color='red', bold=True, bright=True),
#         critical=dict(color='black', bold=True, background='red'),
#     ),
#     field_styles=dict(
#         name=dict(color='white'),
#         levelname=dict(color='white'),
#         asctime=dict(color='white'),
#         funcName=dict(color='white'),
#         lineno=dict(color='white'),
#     )
#     )
#
#     ch = logging.StreamHandler(stream=sys.stdout)
#     ch.setFormatter(fmt=coloredFormatter)
#     logger.addHandler(hdlr=ch)
#
#     logger.setLevel(level=level)   #levels --> DEBUG, INFO, WARNING, ERROR, CRITICAL