# Graphical-Sentiment-Analysis
## Architecture: 
![image](./src/GcnArch.jpeg)
## Graph Convolution Equation: 
$$x_i = \underset{j \epsilon N(i) \cup \{i\}}{\overline{X}} \dfrac{1}{\sqrt{deg(i)}\sqrt{deg(j)}} \tanh(A\,x_j  + B e_{ji}) + Bias$$
