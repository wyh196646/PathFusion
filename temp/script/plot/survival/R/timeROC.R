library(timeROC)
library(survival)

setwd("C:\\Users\\19664\\Desktop\\R")

##################################

select_by_p <- read.csv("C:\\Users\\19664\\Desktop\\R\\formatted_survival_analysis.csv")
head(select_by_p)

ROC <- timeROC(T = select_by_p$survival_time_day,   
               delta = select_by_p$censor_state,   
               marker = select_by_p$risk_score_scale,   
               cause = 1,                
               weighting = "marginal",  
               times = c(365, 365*3, 365*5), 
               iid = TRUE)

# 使用小清新配色，通过RGB颜色定义
colors <- c("#EDC66A", "#9FDAF7", "#C74546")

# 保存为PNG格式
png(filename = 'timeROC.png', height = 800, width = 800, res = 120, family = 'Times')

plot(ROC, time = 365, col = colors[1], lwd = 2, title = "")
plot(ROC, time = 365*3, col = colors[2], add = TRUE, lwd = 2)
plot(ROC, time = 365*5, col = colors[3], add = TRUE, lwd = 2)

# 添加图例
legend("bottomright",
       legend = c(paste0("AUC at 1 year: ", round(ROC$AUC[1], 3)),
                  paste0("AUC at 3 year: ", round(ROC$AUC[2], 3)),
                  paste0("AUC at 5 year: ", round(ROC$AUC[3], 3))),
       col = colors,
       lty = 1, lwd = 2, bty = "n")

dev.off() # 关闭PNG图形设备
