#
# This is R Terminal to execute code and test it in real
# time while CSV File is converted into PMML File
# using R libraries
#

#
#

# Get argument from command
# line when Rterm is executed
args0 <- base::commandArgs()
args0

# Substring to search for
pattern <- "\\\\"

# Looking for last occurrence of "\" (backslash)
matches <- gregexpr(pattern, args0[3])
matched_strings <- regmatches(args0[3], matches)
index <- length(matches[[1]])
last_index <- matches[[1]][index]

#
#

# Get working directory from
# the command line parameter
# ---
# Example: ..\.Rterm -f "..\..\ToPMML.R"
working_directory <- substr(args0[3], start = 1, stop=last_index)

#
#

# Installing R packages

# Define required packages
packages <- c("Rserve", "r2pmml", "rpart")

# Install missing packages
install.packages(setdiff(packages, rownames(installed.packages())))

# Load all packages 
# (packages installation done -- once)
lapply(packages, library, character.only = TRUE)

# -------

# Loading library
library(r2pmml)

# Loading library
library(rpart)

# ---

# Load CSV File into variable
# Change a relative path -- remove absolute path
data <- read.csv(paste(working_directory, "Elnino.csv", sep=""))
data

#
#

# Data Example
#     buoy_day_ID buoy day latitude longitude zon_winds mer_winds humidity airtemp s_s_temp
# 1             1    1   1     8.96   -140.32      -6.3      -6.4    83.50   27.32    27.57
# ...
# ...
# ...
# 781         781   59  13    -8.04    164.82        NA        NA    95.50   28.44    28.51
# 782         782   59  14    -8.04    164.81        NA        NA    93.40   28.67    28.61

#
#

# Load model into variable
model <- rpart(s_s_temp ~ ., data = data)

#
#

# Model variable result example
# n=709 (73 observations deleted due to missingness)
#
#   1) root 709 1361.840000 28.29326  
#   ....
#   ....
#            59) buoy_day_ID>=346.5 122   20.874840 29.22369 *
#        15) airtemp>=28.475 135   16.056080 29.48037 *

#
#

# Converting to PMML File
r2pmml(model, paste(working_directory, "Elnino.pmml", sep=""))

#
#

# Ouput message result
print("")
print("")
print("")

print("Process finished successfully.")
print("CSV converted to PMML format.")

print("")
print("")

#
#

