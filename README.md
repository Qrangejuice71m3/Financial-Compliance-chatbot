# 金融合规 Chatbot（MiFID II + SFDR）

## 功能

- 财务偏好采集状态机（3问）：
  - 风险承受能力（Low/Medium/High）
  - 投资期限（Short/Medium/Long）
  - 亏损承受能力（Low/Medium/High）
- ESG 偏好采集状态机（3问）：
  - 环境、社会、治理目标打分（1-5）
- 自然语言画像抽取：
  - 通过 LLM 从一句话中抽取 financial profile 与 ESG profile，并自动回填 investor_profile
  - 若画像未补全，系统自动追问缺失 slot；补全后会回显识别到的财务与 ESG 分数
- 自然语言问答：
  - 在非问卷状态下，调用 LLM API 回答客户问题
- Trade-off 拦截器：
  - `check_compliance_tension(user_risk_level, selected_product_risk)`
  - 当产品风险高于客户风险承受能力时，输出 MiFID II Art 24/25 合规提示
- 向量检索 + EET 披露回答：
  - 当用户询问基金绿色程度时，系统先在默认 EET 数据中做向量检索，再基于检索结果回答
  - 若 EET 未提及“零排放/净零保证”，回答强制追加免责声明：`该信息基于当前披露，并不代表最终的碳中和保证。`

## 启动

```bash
cd /Users/juiceq/Desktop/金融合规
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 app.py
```

浏览器访问：`http://127.0.0.1:8000/`

## LLM API 配置（可选）

未配置时系统仍可运行，但自然语言问答走 fallback。

```bash
export OPENAI_API_KEY="你的key"
export OPENAI_MODEL="gpt-4o-mini"
# 可选，自定义网关地址
# export OPENAI_BASE_URL="https://api.openai.com/v1/chat/completions"
```

也可以直接在网页中的 `LLM API 配置` 面板填写 `API Key / Base URL / Model`，配置会保存在浏览器本地并随 `/chat` 请求发送。

## 快速测试流程

1. 输入：`我想做投资规划`（触发财务偏好3问）
2. 回答：`中等`、`长期`、`中`
3. 输入：`我关注ESG`（触发 ESG 3问）
4. 回答：`5分`、`4分`、`3分`
5. 输入：`推荐一个高风险绿色产品`（触发 Trade-off 检查）
6. 输入：`这个基金绿色程度怎么样？`（触发 EET 检索）

## 模拟默认 EET 数据

- 文件：`/Users/juiceq/Desktop/金融合规/eet_default.json`
- 含 3 只基金的模拟官方 EET 披露字段（SFDR 分类、Taxonomy 对齐、PAI、治理政策、零排放声明等）
