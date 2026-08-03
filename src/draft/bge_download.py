"""
download_bge_model.py
独立的 BGE 模型下载脚本，下载前指定存放目录
运行一次后，模型永久保存在指定位置，embedder_bge.py 直接加载即可
"""
import os

from huggingface_hub import snapshot_download

from src.logger import get_logger
logger = get_logger(__name__)
# ============================================================
# 第一步：指定模型存放目录（下载前设置）
# ============================================================
# 在这里指定你想存放模型的绝对路径
# 示例：Windows 用正斜杠或双反斜杠
MODEL_DIR = "D:/workspace/models/huggingface/bge-reranker-base"  # ← 改成你要存放的目录

# ============================================================
# 第二步：设置环境变量（让 huggingface 把模型下载到指定位置）
# ============================================================
os.environ['HF_HOME'] = MODEL_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(MODEL_DIR, 'hub')

# 如果你在国内，建议加上镜像源加速下载
# 取消下面这行的注释即可启用
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ============================================================
# 第三步：下载模型
# ============================================================
from sentence_transformers import SentenceTransformer, CrossEncoder

# 需要下载的模型列表
# - "BAAI/bge-small-zh"  : 轻量版，384维，推荐（约400MB）
# - "BAAI/bge-base-zh"   : 标准版，768维（约1.1GB）
# - "BAAI/bge-large-zh"  : 大模型，1024维（约1.3GB）
# MODEL_NAME = "BAAI/bge-small-zh"  # ← 想换模型就改这里

MODEL_NAME = "BAAI/bge-reranker-base"

def download_model():
    logger.info("=" * 60)
    logger.info("📥 BGE 模型下载工具")
    logger.info("=" * 60)
    logger.info(f"   存放目录: {MODEL_DIR}")
    logger.info(f"   模型名称: {MODEL_NAME}")
    logger.info("=" * 60)

    try:
        logger.info("🔨 开始下载模型，请耐心等待...")
        logger.info("   （首次下载需要几分钟，取决于网络速度）")
        print()

        # 这行代码会触发下载，并自动保存到上面指定的目录
        # model = SentenceTransformer(MODEL_NAME)  # 下载 embedding 模型
        # 下载 reranker 模型
        snapshot_download(
            repo_id = MODEL_NAME,
            local_dir = MODEL_DIR,
            local_dir_use_symlinks = False,
            resume_download = True,
        )


        # 下载完成后，做一次简单推理验证

        # logger.info("🧪 验证模型是否可用...")
        # test_vector = model.encode("测试文本")
        # logger.info(f"   ✅ 测试成功！向量维度: {len(test_vector)}")

        # 显示模型保存位置

        logger.info("=" * 60)
        logger.info("✅ 模型下载完成！")
        logger.info("=" * 60)
        logger.info(f"   模型名称: {MODEL_NAME}")
        logger.info(f"   存放位置: {os.path.join(MODEL_DIR, 'hub')}")
        # logger.info(f"   向量维度: {len(test_vector)}")

        logger.info("📝 接下来运行 embedder_bge.py 即可直接加载此模型")
        logger.info("=" * 60)

    except Exception as e:
        logger.info(f"\n❌ 下载失败: {e}")
        logger.info("\n可能的原因：")
        logger.info("   1. 网络连接问题，请检查网络")
        logger.info("   2. 存放目录没有写入权限")
        logger.info("   3. 如果使用镜像源，请确认 HF_ENDPOINT 设置正确")


if __name__ == "__main__":
    download_model()