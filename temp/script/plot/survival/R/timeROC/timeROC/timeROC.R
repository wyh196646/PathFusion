library(timeROC)
library(survival)

setwd("C:\\Users\\yanrui\\Desktop\\timeROC")

##################################

select_by_p = read.csv("C:\\Users\\yanrui\\Desktop\\timeROC\\final_blca.csv")
head(select_by_p)

ROC <- timeROC(T=select_by_p$survival_time_day,   
               delta=select_by_p$censor_state,   
               marker=select_by_p$risk_score_scale,   
               cause=1,                #阳性结局指标数值
               weighting="marginal",   #计算方法，默认为marginal
               times=c(365, 365*3, 365*5),       #时间点，选取1年，3年和5年的生存率
               iid=TRUE)

pdf(file='timeROC.pdf',height=6,width=8, family='Times')

plot(ROC, 
     time=365, col="red", lwd=2, title = "")   #time是时间点，col是线条颜色
plot(ROC,
     time=365*3, col="blue", add=TRUE, lwd=2)    #add指是否添加在上一张图中
plot(ROC,
     time=365*5, col="orange", add=TRUE, lwd=2)

#添加标签信息
legend("bottomright",
       c(paste0("AUC at 1 year: ",round(ROC[["AUC"]][1],3)), 
         paste0("AUC at 3 year: ",round(ROC[["AUC"]][2],3)), 
         paste0("AUC at 5 year: ",round(ROC[["AUC"]][3],3))),
       col=c("red", "blue", "orange"),
       lty=1, lwd=2,bty = "n") 

dev.off()#关闭PDF

