library(kit)
source("global.R")

createFile <- function(name, content){
  fileConn<-file(name)
  writeLines(content, fileConn)
  close(fileConn)
}

getRandomN <- function(maximum, minimum){
  return( ceiling(runif(n=1, min=minimum, max=maximum)) )  
}

getCurrentYear <- function(){
  return(format(Sys.Date(), "%Y"))
}

createNumber <- function(len, leading_zero=TRUE) {
  retval <- ""
  
  if (leading_zero){
    zero <- ""  
    
    for(x in 1:len-1){ zero <- c(zero, "0")  }
    
    retval <- paste( c(paste(zero, collapse=""), getRandomN(100, 10)), collapse="" )
    
    retval <- substring(retval, nchar(retval) - (len-1), nchar(retval))
  }
  else {
    num <- ""
    for(x in 1:len){ num <- c(num, getRandomN(9, 1)) }
    retval <- paste(num, collapse="");
  }
  
  return(retval)  
}

createAlpha <- function(len){
  retval <- ""
  for (x in 1:len) {
    chr <- intToUtf8(getRandomN(90, 65))
    retval <- c(retval, chr)
  }

  return(paste(retval, collapse=""))
}

createLine <- function(rowid) {
  setConfiguration()
  
  id <- rowid
  
  prefix <- createNumber(4)
  prefix <- substring(prefix, nchar(prefix) - bike[["prefix_len"]], nchar(prefix))

  suffix <- createAlpha(bike[["suffix_len"]])

  bike_model<- paste(c(prefix, suffix), collapse="-")
  quantity <- getRandomN(4, 1)
  

  h <- 9 + getRandomN(10, 0)
  hh <- paste(c("0", iif(h > 12, h - 12, h)), collapse="")
  mm <- paste(c("0", iif(h != 19, getRandomN(59, 0), 0)), collapse="")
  rr <- iif(as.numeric(hh) == 12 || (as.numeric(hh) >= 1 && as.numeric(hh) <= 7), "PM", "AM")
  
  hh <- substring(hh, nchar(hh) - hhmm[["hour_len"]], nchar(hh))
  mm <- substring(mm, nchar(mm) - hhmm[["minute_len"]], nchar(mm))
  
  hour <- paste(c(hh, ":", mm, rr), collapse="")    

  date <- paste(c(getCurrentYear(), "-01-01"), collapse="")
  
  from_to <- ""
  if (as.numeric(hh) >= 9 && as.numeric(hh) < 10) { from_to <- label[["first"]] }
  if (as.numeric(hh) >= 10 && as.numeric(hh) < 12) { from_to <- label[["second"]] }
  if (as.numeric(hh) == 12 || (as.numeric(hh) >= 1 && as.numeric(hh) < 2)) { from_to <- label[["third"]] }
  if (as.numeric(hh) >= 2 && as.numeric(hh) < 4) { from_to <- label[["fourth"]] }
  if (as.numeric(hh) >= 4 && as.numeric(hh) < 6) { from_to <- label[["fifth"]] }
  if (as.numeric(hh) >= 6 && as.numeric(hh) <= 7) { from_to <- label[["sixth"]] }
  
  shift <- ""
  if (from_to == label[["first"]]) { shift <- label[["morning"]] }
  if (from_to == label[["second"]]) { shift <- label[["morning"]] }
  if (from_to == label[["third"]]) { shift <- label[["midday"]] }
  if (from_to == label[["fourth"]]) { shift <- label[["afternoon"]] }
  if (from_to == label[["fifth"]]) { shift <- label[["afternoon"]] }
  if (from_to == label[["sixth"]]) { shift <- label[["evening"]] }
  
  weather <- dict_weather[[sprintf("%d",getRandomN(3, 1))]]
  season <- dict_season[[sprintf("%d",getRandomN(4, 1))]]
  
  line <- paste0(
    id,
    delimiter,
    bike_model,
    delimiter,
    quantity, 
    delimiter,
    hour,
    delimiter,
    date,
    delimiter,
    from_to, 
    delimiter,
    shift, 
    delimiter,
    weather, 
    delimiter,
    season
  )
  
  return(line)
}

createModel <- function(name) {
  lines <- list()
  line <- ""
  
  for (i in 1:49) {
    line <- createLine(i)
    lines <- append(lines, line)
  }
  
  createFile(name, paste0(lines))
}

