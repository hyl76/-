# 在开头导入 shap
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt



# 修改页面标题
st.set_page_config(page_title="肺癌早期筛查风险预测系统", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('xgboost_model.pkl')

model = load_model()

# 修改主标题
st.title("肺癌早期筛查风险预测系统")
st.write("请输入患者信息进行预测")

with st.form("prediction_form"):
    # First row: Basic Information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender = st.selectbox("性别", 
                            options=[0, 1],
                            format_func=lambda x: "女" if x == 0 else "男")
    with col2:
        age = st.selectbox("年龄", 
                          options=[0, 1, 2, 3, 4, 5],
                          format_func=lambda x: {0:"40-49岁", 1:"50-54岁", 2:"55-59岁", 
                                               3:"60-64岁", 4:"65-69岁", 5:"70-74岁"}[x])
    with col3:
        education = st.selectbox("文化程度",
                               options=[1, 2, 3, 4, 5, 6],
                               format_func=lambda x: {1:"未受教育", 2:"小学", 
                                                    3:"初中", 4:"高中/中专/技校", 
                                                    5:"大专", 6:"大学及以上"}[x])
    with col4:
        occupation = st.selectbox("职业暴露",
                                options=[0, 1],
                                format_func=lambda x: "否" if x == 0 else "是")

    # Second row: Living Habits
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        vegetables = st.selectbox("新鲜蔬菜",
                                options=[0, 1, 2],
                                format_func=lambda x: {0:"从不", 1:"<5斤/周", 
                                                     2:"≥5斤/周"}[x])
    with col6:
        fruits = st.selectbox("新鲜水果",
                            options=[0, 1, 2],
                            format_func=lambda x: {0:"从不", 1:"<2.5斤/周", 
                                                 2:"≥2.5斤/周"}[x])
    with col7:
        meat = st.selectbox("畜肉",
                          options=[0, 1, 2],
                          format_func=lambda x: {0:"从不", 1:"≤7两/周", 
                                               2:">7两/周"}[x])
    with col8:
        grains = st.selectbox("粗粮",
                            options=[0, 1, 2],
                            format_func=lambda x: {0:"从不", 1:"<1斤/周", 
                                                 2:"≥1斤/周"}[x])

    # Third row: Dietary Preferences
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        taste = st.selectbox("口味",
                           options=[1, 2, 3],
                           format_func=lambda x: {1:"重盐", 2:"适中", 3:"清淡"}[x])
    with col10:
        oil = st.selectbox("油脂",
                         options=[1, 2, 3],
                         format_func=lambda x: {1:"较高", 2:"适中", 3:"较低"}[x])
    with col11:
        preserved_food = st.selectbox("腌晒食品",
                                    options=[1, 2, 3],
                                    format_func=lambda x: {1:"从不", 2:"有时", 
                                                         3:"经常"}[x])
    with col12:
        air_pollution = st.selectbox("空气污染",
                                   options=[0, 1],
                                   format_func=lambda x: "否" if x == 0 else "是")

    # Fourth row: Smoking and Drinking
    col13, col14, col15, col16 = st.columns(4)
    with col13:
        smoking = st.selectbox("吸烟状况",
                             options=[0, 1, 2],
                             format_func=lambda x: {0:"从不吸", 1:"目前仍在吸", 
                                                  2:"目前已戒烟"}[x])
    with col14:
        indoor_smoking = st.selectbox("室内吸烟",
                                    options=[0, 1],
                                    format_func=lambda x: "否" if x == 0 else "是")
    with col15:
        drinking = st.selectbox("饮酒频率",
                              options=[0, 1, 2],
                              format_func=lambda x: {0:"从不饮", 1:"目前仍常饮", 
                                                   2:"目前已戒酒"}[x])
    with col16:
        tea = st.selectbox("饮茶频率",
                          options=[0, 1, 2],
                          format_func=lambda x: {0:"从不饮", 1:"目前仍常饮", 
                                               2:"目前不常饮"}[x])

    # Fifth row: Medical History
    col17, col18, col19, col20 = st.columns(4)
    with col17:
        bronchitis = st.selectbox("慢性支气管炎",
                                options=[0, 1],
                                format_func=lambda x: "否" if x == 0 else "是")
    with col18:
        emphysema = st.selectbox("肺气肿",
                               options=[0, 1],
                               format_func=lambda x: "否" if x == 0 else "是")
    with col19:
        tuberculosis = st.selectbox("肺结核",
                                  options=[0, 1],
                                  format_func=lambda x: "否" if x == 0 else "是")
    with col20:
        family_history = st.selectbox("肺癌家族史",
                                    options=[0, 1],
                                    format_func=lambda x: "否" if x == 0 else "是")

    # Sixth row: Additional Factors
    col21, col22, _, _ = st.columns(4)
    with col21:
        exercise = st.selectbox("经常体育锻炼",
                              options=[0, 1],
                              format_func=lambda x: "否" if x == 0 else "是")
    with col22:
        cooking_fume = st.selectbox("做饭油烟",
                                  options=[1, 2, 3, 4],
                                  format_func=lambda x: {1:"无烟", 2:"少许", 
                                                       3:"较多", 4:"很多"}[x])

    submitted = st.form_submit_button("预测", type="primary")

    if submitted:
        input_data = np.array([[
            gender, age, education, occupation, vegetables, fruits, meat, 
            grains, taste, oil, preserved_food, air_pollution, smoking,
            indoor_smoking, drinking, tea, exercise, cooking_fume,
            tuberculosis, bronchitis, emphysema, family_history
        ]])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        st.markdown("---")
        # st.subheader("预测结果")

        risk_prob = probability[0][1]
        optimal_threshold = 0.1992
        
        if risk_prob >= optimal_threshold:
            st.error("# 预测结果：肺癌高风险人群")
            st.markdown(f"## **风险概率：{risk_prob:.1%}**")
        else:
            st.success("# 预测结果：肺癌低风险人群")
            st.markdown(f"## **风险概率：{risk_prob:.1%}**")
            
        # # 添加主要影响因素分析
        # st.markdown("---")
        # st.subheader("主要影响因素")
        
        # # 计算SHAP值
        # explainer = shap.TreeExplainer(model)
        # shap_values = explainer.shap_values(input_data)
        
        # # 特征名称列表
        # feature_names = [
        #     "性别", "年龄", "文化程度", "职业暴露", 
        #     "新鲜蔬菜", "新鲜水果", "畜肉", "粗粮",
        #     "口味", "油脂", "腌晒食品", "空气污染",
        #     "吸烟状况", "室内吸烟", "饮酒频率", "饮茶频率",
        #     "体育锻炼", "做饭油烟", "肺结核", "慢性支气管炎",
        #     "肺气肿", "肺癌家族史"
        # ]
        
        # # 获取SHAP值
        # shap_vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        
        # # 获取前5个最重要的特征及其影响
        # feature_importance = [(abs(val), name, val) for val, name in zip(shap_vals, feature_names)]
        # top_features = sorted(feature_importance, reverse=True)[:5]
        
        # # 创建选项值的映射字典
        # value_mapping = {
        #     "性别": "女" if gender == 0 else "男",
        #     "年龄": {0:"40-49岁", 1:"50-54岁", 2:"55-59岁", 3:"60-64岁", 4:"65-69岁", 5:"70-74岁"}[age],
        #     "文化程度": {1:"未受教育", 2:"小学", 3:"初中", 4:"高中/中专/技校", 5:"大专", 6:"大学及以上"}[education],
        #     "职业暴露": "否" if occupation == 0 else "是",
        #     "新鲜蔬菜": {0:"从不", 1:"<5斤/周", 2:"≥5斤/周"}[vegetables],
        #     "新鲜水果": {0:"从不", 1:"<2.5斤/周", 2:"≥2.5斤/周"}[fruits],
        #     "畜肉": {0:"从不", 1:"≤7两/周", 2:">7两/周"}[meat],
        #     "粗粮": {0:"从不", 1:"<1斤/周", 2:"≥1斤/周"}[grains],
        #     "口味": {1:"重盐", 2:"适中", 3:"清淡"}[taste],
        #     "油脂": {1:"较高", 2:"适中", 3:"较低"}[oil],
        #     "腌晒食品": {1:"从不", 2:"有时", 3:"经常"}[preserved_food],
        #     "空气污染": "否" if air_pollution == 0 else "是",
        #     "吸烟状况": {0:"从不吸", 1:"目前仍在吸", 2:"目前已戒烟"}[smoking],
        #     "室内吸烟": "否" if indoor_smoking == 0 else "是",
        #     "饮酒频率": {0:"从不饮", 1:"目前仍常饮", 2:"目前已戒酒"}[drinking],
        #     "饮茶频率": {0:"从不饮", 1:"目前仍常饮", 2:"目前不常饮"}[tea],
        #     "体育锻炼": "否" if exercise == 0 else "是",
        #     "做饭油烟": {1:"无烟", 2:"少许", 3:"较多", 4:"很多"}[cooking_fume],
        #     "肺结核": "否" if tuberculosis == 0 else "是",
        #     "慢性支气管炎": "否" if bronchitis == 0 else "是",
        #     "肺气肿": "否" if emphysema == 0 else "是",
        #     "肺癌家族史": "否" if family_history == 0 else "是"
        # }
        
        # # 显示主要影响因素
        # for abs_val, name, val in top_features:
        #     impact = "增加" if val > 0 else "降低"
        #     selected_value = value_mapping[name]
        #     st.write(f"• {name}（{selected_value}）: {impact}风险 ({abs(val):.4f})")
            
        # st.info("注：以上显示了对预测结果影响最大的5个因素，正值表示增加风险，负值表示降低风险。")


# 侧边栏信息
st.sidebar.title("关于")
st.sidebar.info(
    "这是一个基于XGBoost的肺癌风险预测系统。\n\n"
    "系统从基本信息、生活习惯、饮食习惯和病史等多个维度分析，"
    "预测肺癌风险。\n\n"
    "预测结果仅供参考，请以医生建议为准。"
)
