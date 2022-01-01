library(ggplot2)

iterations=seq(0,50000,1)
gamma=c()
for(i in time){
  if(i<50){
    gamma=c(gamma,0)
  }
  else if(i<200){
    gamma=c(gamma,200**(-0.6))
  }
  else{
    gamma=c(gamma,i**(-0.6))
  }
}
df=as.data.frame(cbind(iterations,gamma))
ggplot()+geom_point(data=df[1:2500,],aes(x=iterations,y=gamma),color="blue")

setwd("C:\\Users\\Akshay\\Research Project")

library(reshape2)
h<-function(str){
  df1<-read.csv(str,header=TRUE)
  df1[,"Iterations"]=seq(0,50000,1)
  mdf1<-melt(df1,id="Iterations")
  ggplot()+geom_line(data=mdf1,aes(x=Iterations,y=value,color=variable),size=1)+
    theme(legend.key.size = unit(2, 'cm'), #change legend key size
          legend.key.height = unit(1, 'cm'), #change legend key height
          legend.key.width = unit(2, 'cm'), #change legend key width
          legend.title = element_text(size=14), #change legend title font size
          legend.text = element_text(size=15),#change legend text font size
          legend.position = c(0.8, 0.2))+labs(color='Number of Particles')+
    theme(axis.text.x = element_text(color = "grey20", size = 15, angle = 0, hjust = 1, vjust = 0, face = "plain"),
          axis.text.y = element_text(color = "grey20", size = 15, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
          axis.title.x = element_text(color = "grey20", size = 12, angle = 0, hjust = .5, vjust = 0, face = "plain"),
          axis.title.y = element_text(color = "grey20", size = 12, angle = 90, hjust = .5, vjust = .5, face = "plain"))
  
}

h("Dependency of phi with N-Normal parametrization.csv")
h("Dependency of sigma with N-Normal parametrization.csv")
h("Dependency of Beta with N-Normal parametrization.csv")

h("Dependency of phi with N-Log parametrization.csv")
h("Dependency of sigma with N-Log parametrization.csv")
h("Dependency of Beta with N-Log parametrization.csv")

g<-function(str){
  df1<-read.csv(str,header=TRUE)
  df1[,"Iterations"]=seq(0,150000,1)
  mdf1<-melt(df1,id="Iterations")
  ggplot()+geom_line(data=mdf1,aes(x=Iterations,y=value,color=variable),size=1, show.legend = FALSE)+
    theme(legend.key.size = unit(2, 'cm'), #change legend key size
          legend.key.height = unit(1, 'cm'), #change legend key height
          legend.key.width = unit(2, 'cm'), #change legend key width
          legend.title = element_text(size=14), #change legend title font size
          legend.text = element_text(size=15),#change legend text font size
          legend.position = c(0.8, 0.2))+labs(color='Starts')+
    theme(axis.text.x = element_text(color = "grey20", size = 15, angle = 0, hjust = 1, vjust = 0, face = "plain"),
          axis.text.y = element_text(color = "grey20", size = 15, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
          axis.title.x = element_text(color = "grey20", size = 12, angle = 0, hjust = .5, vjust = 0, face = "plain"),
          axis.title.y = element_text(color = "grey20", size = 12, angle = 90, hjust = .5, vjust = .5, face = "plain"))
  
}

g("Phi-SV-Starts.csv")
g("Sigma-SV-Starts.csv")
g("Beta-SV-Starts.csv")

h("LGSS-C-Particles.csv")
g("LGSS-C-Starts.csv")

g("LGSS-high-C.csv")





