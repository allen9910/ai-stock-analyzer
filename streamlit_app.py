import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import akshare as ak
import pandas as pd
import re
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI 热点个股研判终端", layout="wide")

# ================= 全量关键词（你提供的 + 补充）=================
KEYWORDS = [
    # AI & 算力
    "AI", "人工智能", "大模型", "GLM", "GLM-Image", "智谱", "昇腾", "Atlas", "MindSpore",
    "算力", "GPU", "NPU", "AI芯片", "寒武纪", "海光信息", "壁仞", "天数", "燧原",
    "AI服务器", "训练集群", "推理", "国产算力", "全栈自主", "SOTA", "Hugging Face",
    
    # 半导体
    "半导体", "芯片", "光刻机", "ASML", "中芯国际", "华虹", "华为海思", "长江存储", "长鑫存储",
    "先进封装", "Chiplet", "HBM", "EDA", "Synopsys", "Cadence", "华大九天", "概伦电子",
    "晶圆", "设备", "材料", "刻蚀", "薄膜", "离子注入", "北方华创", "中微公司", "拓荆科技", "盛美上海",
    
    # 光通信
    "光模块", "CPO", "800G", "1.6T", "LPO", "硅光", "Coherent", "新易盛", "中际旭创", "天孚通信",
    
    # 新能源
    "光伏", "风电", "氢能", "储能", "锂电池", "固态电池", "钠离子", "宁德时代", "比亚迪", "亿纬锂能",
    "隆基", "通威", "晶科", "逆变器", "HJT", "TOPCon", "钙钛矿", "BC电池",
    
    # 新材料
    "玻纤", "低介电", "覆铜板", "PCB", "芳纶", "PI膜", "碳纤维", "稀土", "高温合金", "超导",
    
    # 数字经济
    "5G", "6G", "卫星互联网", "东数西算", "信创", "国产替代", "操作系统", "数据库",
    "华为欧拉", "openGauss", "达梦", "人大金仓", "麒麟软件",
    
    # 政策与主题
    "新质生产力", "设备更新", "以旧换新", "专精特新", "小巨人", "科创板", "北交所",
    "并购重组", "回购", "增持", "减持", "定增", "股权激励",
    
    # 市场情绪与博弈
    "击鼓传花", "接盘", "游资", "机构", "量化", "散户", "龙虎榜", "涨停", "连板", "断板",
    "预期差", "兑现", "利好出尽", "分歧", "一致", "高潮", "退潮", "卡位", "造梦", "故事",
    "落地性", "订单验证", "量产", "良率", "毛利率", "净利率",
    
    # 宏观经济
    "降息", "降准", "CPI", "PPI", "社融", "PMI", "美联储", "人民币", "国债", "汇率",
    
    # 前沿科技
    "低空经济", "eVTOL", "飞行汽车", "亿航", "小鹏汇天", "峰飞",
    "商业航天", "火箭", "卫星", "银河航天", "时空道宇",
    "脑机接口", "Neuralink", "侵入式", "非侵入式",
    "量子计算", "量子通信", "本源量子", "国盾量子",
    "人形机器人", "具身智能", "特斯拉Optimus", "宇树", "优必选"
]

# ================= 获取 AkShare 新闻（自动抓取，近5天）=================
@st.cache_data(ttl=600)
def fetch_akshare_news():
    try:
        df = ak.stock_news_em(symbol="全部")
        if df.empty:
            return pd.DataFrame()
        df['时间戳'] = pd.to_datetime(df['发布时间'], errors='coerce')
        df = df.dropna(subset=['时间戳'])
        df['发布日期'] = df['时间戳'].dt.strftime('%Y-%m-%d')
        df['发布时间'] = df['时间戳'].dt.strftime('%H:%M')
        df = df.rename(columns={'新闻标题': '标题', '新闻内容': '内容'})
        # 仅保留最近5天新闻
        cutoff = datetime.now() - timedelta(days=5)
        df = df[df['时间戳'] >= cutoff]
        df = df.sort_values('时间戳', ascending=False).reset_index(drop=True)
        return df[['标题', '内容', '发布日期', '发布时间']].head(50)
    except Exception as e:
        st.error(f"AkShare 获取失败: {e}")
        return pd.DataFrame()

# ================= 主应用 =================
def main():
    st.title("🔍 AI 热点个股研判终端")

    # === 顶部关键词搜索框 ===
    user_keywords = st.text_input(
        "🔎 输入关键词（如：低空经济, 昇腾, 设备更新）",
        placeholder="支持多个关键词，用中文逗号或空格分隔",
        key="keyword_input"
    )

    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        st.error("❌ 请在 .env 文件中配置 ZHIPU_API_KEY")
        return

    # === 获取原始新闻 ===
    raw_news_df = fetch_akshare_news()
    if raw_news_df.empty:
        st.warning("暂无新闻数据，请稍后再试。")
        return

    # === 处理关键词过滤 ===
    filtered_news = raw_news_df.copy()
    keywords_list = []
    if user_keywords.strip():
        # 分割关键词（支持中文逗号、英文逗号、空格）
        keywords_list = [kw.strip() for kw in re.split(r'[,\s，]+', user_keywords.strip()) if kw.strip()]
        if keywords_list:
            def contains_keyword(text):
                return any(kw in text for kw in keywords_list)
            mask = filtered_news['标题'].apply(contains_keyword) | filtered_news['内容'].apply(contains_keyword)
            filtered_news = filtered_news[mask].reset_index(drop=True)

    if filtered_news.empty:
        st.info("未找到匹配关键词的新闻，请尝试其他关键词。")
        return

    # === 状态管理 ===
    if 'selected_idx' not in st.session_state:
        st.session_state.selected_idx = 0
        st.session_state.analysis_cache = {}

    col_list, col_detail = st.columns([2.8, 7.2])

    with col_list:
        st.subheader(f"📰 新闻列表（共 {len(filtered_news)} 条）")
        if keywords_list:
            st.caption(f"关键词：{'、'.join(keywords_list)}")
        for idx, row in filtered_news.iterrows():
            is_selected = idx == st.session_state.selected_idx
            if st.button(
                f"**{row['标题']}**\n`{row['发布日期']} {row['发布时间']}`",
                key=f"news_{idx}",
                type="primary" if is_selected else "secondary",
                use_container_width=True
            ):
                st.session_state.selected_idx = idx
                st.rerun()

    with col_detail:
        current = filtered_news.iloc[st.session_state.selected_idx]
        cache_key = f"{current['标题']}|{user_keywords}"

        # 显示新闻详情
        st.markdown("### 📌 新闻详情")
        st.caption(f"{current['发布日期']} {current['发布时间']}")
        
        # 高亮关键词
        content_display = current['内容']
        if keywords_list:
            for kw in keywords_list:
                content_display = re.sub(
                    f"({re.escape(kw)})",
                    r"<mark style='background:#fffacd;font-weight:bold'>\1</mark>",
                    content_display,
                    flags=re.IGNORECASE
                )
        
        st.markdown(
            f"<div style='background:#f9fafb;padding:14px;border-radius:10px;margin-bottom:24px;'>{content_display}</div>",
            unsafe_allow_html=True
        )

        # 缓存分析结果
        if cache_key not in st.session_state.analysis_cache:
            with st.spinner("🧠 AI 正在分析受益股及三维度阶段..."):
                try:
                    llm = ChatOpenAI(
                        api_key=api_key,
                        base_url="https://open.bigmodel.cn/api/paas/v4/",
                        model="glm-4",
                        temperature=0.2
                    )
                    prompt = ChatPromptTemplate.from_messages([
                        ("system",
                         "你是顶级产业资本操盘手。请严格按以下步骤处理：\n\n"
                         "1️⃣ **先列出 2-4 只最直接受益的 A 股**，格式必须为：**公司全称（6位数字代码）**\n"
                         "   - 必须是真实存在的 A 股（代码以 00/30/60/68 开头）\n"
                         "   - 禁止虚构公司或使用港股/美股代码\n\n"
                         "2️⃣ **对每只股票，分别进行三维度研判**：\n"
                         "   - **卡位**：是否真实卡位？有无技术/订单/政策壁垒？是否蹭概念？\n"
                         "   - **预期差**：当前市场预期 vs 潜在空间（可用%估算），是否存在认知差？\n"
                         "   - **击鼓传花阶段**：启动 / 加速 / 高潮 / 退潮？主导资金是谁（游资/机构/散户）？\n\n"
                         "3️⃣ 语言犀利、数据化，禁止模糊词。"
                        ),
                        ("user", f"新闻标题：{current['标题']}\n\n新闻内容：{current['内容']}")
                    ])
                    result = (prompt | llm | StrOutputParser()).invoke({})
                    st.session_state.analysis_cache[cache_key] = result
                except Exception as e:
                    st.session_state.analysis_cache[cache_key] = f"❌ 分析失败：{str(e)}"

        # 显示 AI 分析结果
        st.markdown("### 🔍 AI 动态研判结果")
        st.markdown(st.session_state.analysis_cache[cache_key])

if __name__ == "__main__":
    main()
