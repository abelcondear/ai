###### Challenge #####
#
# Your challenge will be to conduct a time series data analysis 
# to forecast future rental bike demands. To do this, you will load, 
# clean, process, analyze, and visualize data. 
#
# You will also pose questions, and seek to answer them meaningfully using 
# the dataset provided. In addition, you will build time series models to 
# forecast future demands.
#
# In this project, we'll use a data set containing the hourly and daily count 
# of rental bikes from year 2025 in the Capital bike share 
# system in Washington, DC, with the corresponding weather and seasonal 
# information.
#
# After you perform your analysis, you will share your findings.
#
# ################## #

# Load required libraries
library(forecast)
library(data.table)
library(clock)
library(lubridate)
library(processx)

source("model.R")

#R version
createModel("input.csv")

#node version
#run("node", c("model.js"))

dd <- fread("input.csv");
items <- dd[,V3]

png(file="rentBike-picture.png")

mts <- ts(items, start = decimal_date(ymd("2025-01-01")), 
        frequency = 365.25/1) #daily frequency

plot(mts, xlab="Daily Data",
     ylab="Amount",
     main="Renting bikes during this year",
     col.main="black")

dev.off()
