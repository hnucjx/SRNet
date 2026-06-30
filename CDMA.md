```mermaid
flowchart TB
    %% 科研风格样式定义
    classDef startend fill:#f0f5f9,stroke:#333,stroke-width:1px,rounded:10px;
    classDef embedproc fill:#e8f0fe,stroke:#2b5dac,stroke-width:1px;
    classDef extractproc fill:#eef6f0,stroke:#3a7d44,stroke-width:1px;
    classDef decision fill:#fff3cd,stroke:#d39e00,stroke-width:1px;
    classDef note fill:#f8f9fa,stroke:#6c757d,stroke-width:1px,stroke-dasharray: 5 5,font-size:10pt;

    %% -------------------------- 水印嵌入阶段 --------------------------
    subgraph 第一阶段:水印编码与CDMA扩频嵌入
        direction TB
        %% 输入
        M[/"原始二进制水印版权信息<br/>$M \\in \\{0,1\\}^L$"/]:::startend
        W[/"训练完成的工业AI模型<br/>原始权重$W$"/]:::startend

        %% 水印编码子模块
        subgraph 水印比特编码（RS编码+二极性映射）
            direction TB
            S1[1. 水印分组<br/>将$M$均分为$n$组，每组长度$P$<br/>$M=\\{m_i\\}_{i=1}^n$]:::embedproc
            S2[2. Reed-Solomon纠错编码<br/>对每个分组做RS编码提升抗干扰能力<br/>得到$M^{RS}=\\{m_i^{RS}\\}_{i=1}^n$]:::embedproc
            S3[3. 二极性映射<br/>比特0→+1、比特1→-1<br/>得到双极性序列$M^{bipolar}=\\{m_i^{bipolar}\\}_{i=1}^n$<br/>$m_i^{bipolar} \\in \\{-1,1\\}^P$]:::embedproc
        end
        N1[/"特性:双极性序列统计均值为0<br/>最小化对模型原始任务性能的干扰"/]:::note
        S3 --> N1

        %% 权重预处理子模块
        SA[权重预处理:随机分组<br/>将$W$分为$n$组，每组含$R$个参数<br/>1组权重对应嵌入1组水印]:::embedproc

        %% CDMA扩频嵌入子模块
        subgraph CDMA扩频嵌入
            direction TB
            S4[4. 生成独立伪随机扩频码<br/>为每比特水印生成长度$R$的±1扩频向量$c$<br/>同组扩频码组成$R \\times P$编码矩阵$C$]:::embedproc
            N4[/"特性:扩频将水印窄带信号扩展至极宽频带<br/>对水印移除攻击高鲁棒性，容量优于传统方案"/]:::note
            S5[5. 计算水印嵌入信号<br/>单比特信号:$S = \\gamma \\cdot b \\cdot c$（$\\gamma$为嵌入强度）<br/>组级信号：$\\gamma \\cdot m_i^{bipolar} \\cdot C$]:::embedproc
            S6[6. 叠加嵌入水印<br/>含水印权重:$W_{wtm} = W + \\gamma \\cdot m_i \\cdot C$]:::embedproc
        end
        N2[/"特性:无需修改原模型训练流程<br/>训练后直接嵌入，低开销、适配大参数量工业模型"/]:::note
        S6 --> N2

        %% 嵌入流程连接
        M --> S1 --> S2 --> S3
        W --> SA
        S3 & SA --> S4
        S4 --> N4
        S4 --> S5 --> S6
        Wwm[/"输出:含水印工业AI模型"/]:::startend
        S6 --> Wwm
    end
    style 第一阶段:水印编码与CDMA扩频嵌入 fill:#f0f7ff,stroke:#2b5dac,stroke-width:1.5px,rounded:10px

    %% -------------------------- 水印提取阶段 --------------------------
    subgraph 第二阶段:CDMA解扩盲水印提取
        direction TB
        %% 提取输入
        Wwtm[/"输入:含水印模型权重$W_{wtm}$<br/>+ 嵌入时权重分组规则<br/>+ 对应伪随机扩频码集"/]:::startend
        N3[/"特性:盲提取无需原始无水印模型<br/>未获取分组规则/扩频码无法提取，安全性高"/]:::note
        Wwtm --> N3

        %% 解扩判决子模块
        subgraph CDMA解扩与比特判决
            direction TB
            S7[7. 权重对齐分组<br/>按嵌入相同规则将$W_{wtm}$分为$n$组，每组$R$个参数]:::extractproc
            S8[8. 解扩相关计算<br/>每比特解扩值：$y_i = c^T \\cdot W_{wtm}^{(i)}$<br/>$W_{wtm}^{(i)}$为对应分组权重]:::extractproc
            D9{9. 比特判决<br/>$y_i > 0$?}:::decision
            B0[判定比特$b=0$]:::extractproc
            B1[判定比特$b=1$]:::extractproc
            Seq[得到完整解扩二进制水印序列]:::extractproc
        end

        %% 解码重构子模块
        subgraph RS解码与水印重构
            direction TB
            S10[10. Reed-Solomon纠错解码<br/>按分组做RS解码，纠正攻击引入的错误，恢复水印分组]:::extractproc
            S11[11. 分组拼接重构<br/>拼接$n$个解码分组，得到完整水印]:::extractproc
            Mout[/"输出：提取的原始水印版权信息$M$"/]:::startend
        end

        %% 提取流程连接
        Wwtm --> S7 --> S8 --> D9
        D9 -- 是 --> B0 --> Seq
        D9 -- 否 --> B1 --> Seq
        Seq --> S10 --> S11 --> Mout
    end
    style 第二阶段:CDMA解扩盲水印提取 fill:#f0fff4,stroke:#3a7d44,stroke-width:1.5px,rounded:10px
```

    %% 阶段间连接
    Wwm --> Wwtm
