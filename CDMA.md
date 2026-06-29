```mermaid
graph TD
    subgraph 水印生成与嵌入
        A[原始水印版权信息] --> B[Reed-Solomon编码]
        B --> C[双极性映射<br/>0→-1, 1→1]
        D[模型权重参数] --> E[模型权重分块]
        E --> F[生成CDMA扩频序列<br/>伪随机序列]
        C --> G[水印信号与扩频序列相乘]
        F --> G
        G --> H[叠加嵌入到模型权重<br/>w' = w + α·s·c]
        H --> I[嵌入水印后的工业模型]
    end
    
    subgraph 盲水印提取
        J[待检测工业模型] --> K[获取模型权重分块方式]
        K --> L[使用对应CDMA扩频序列]
        L --> M[相关运算提取<br/>r = Σ(w'·c)/N]
        M --> N{提取值判断}
        N -->|r < 0| O[水印位 = 0]
        N -->|r ≥ 0| P[水印位 = 1]
        O --> Q[Reed-Solomon解码]
        P --> Q
        Q --> R[拼接水印块]
        R --> S[原始水印版权信息]
    end
    
    I -.-> J
    
    style A fill:#e1f5fe
    style S fill:#c8e6c9
    style I fill:#fff3e0
    style H fill:#fce4ec
    style M fill:#fce4ec
```
