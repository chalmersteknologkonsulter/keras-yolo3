library(readr)

appendCols<-function(duplicates,idx,k){
  x<-imgnames[k,]
  tryCatch({
    if(duplicates[idx] & idx<=length(duplicates))
      x<-appendCols(duplicates,idx+1,k)
  }, error =function(e){
    print(idx)
  }, finally= {})
  
  
  return(cbind(x,anno[idx-1,2:6]))
}


annotering_yoloformat <- read_delim("C:/Users/sebbe/Desktop/drone_anno.csv", 
                                    ";", escape_double = FALSE, trim_ws = TRUE)
View(annotering_yoloformat)

dubbles<-duplicated(annotering_yoloformat[,1])



anno<-annotering_yoloformat


anno[,4]<-anno[,4]+anno[,2]


anno[,5]<-anno[,3]+anno[,5]

anno[,6]<-anno[,6]-2*anno[,6]

imgnames<-unique(anno[,1])

hej<-list()

uniqueImg<-!duplicated(anno[,1])
k=1
for(i in 1:length(uniqueImg)){
  if(uniqueImg[i]){
    hej[[k]]<-data.frame(appendCols(dubbles,i+1,k))
    k=k+1
  }
}



#lapply(hej, write, "test3.txt", append=T, ncolumns=1000 )
sink("last_drone_annotaions.txt")
writeLines(unlist(lapply(hej, paste, sep = ",",collapse=",")))
sink()
