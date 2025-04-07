import logging

# 配置日志的基本设置
# level=logging.INFO: 设置日志级别为INFO，将会记录INFO及以上级别的日志
# format: 设置日志输出格式，包含以下信息：
#   %(asctime)s: 时间戳
#   %(name)s: 记录器名称
#   %(levelname)s: 日志级别
#   %(message)s: 日志消息内容
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 创建一个与当前模块同名的logger实例
# __name__变量将会是该模块的名称
logger = logging.getLogger(__name__)