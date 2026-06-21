#!/usr/bin/env Rscript

#
# This R script file converts CSV File 
# into PMML File using R libraries
#

#
#

# Get argument from command
# line when Rterm is executed
args0 <- base::commandArgs()
args0

os_version <- R.version$os
os_version

os_linux <- "linux"
os_windows <- "mingw"

matches_for_linux <- gregexpr(os_linux, os_version)
matches_for_linux 
is_linux <- ifelse(matches_for_linux[[1]] != -1, TRUE, FALSE)
is_linux

#q()

matches_for_windows <- gregexpr(os_windows, os_version)
matches_for_windows
is_windows <- ifelse(matches_for_windows[[1]] != -1, TRUE, FALSE)
is_windows

#q()

pattern <- ""

matches <- ""
matched_strings <- ""

index <- 0
last_index <- 0

working_directory <- ""

if (is_linux) 
{
	# Substring to search for
	pattern <- "/" #linux or mac character

	# Looking for last occurrence of "\" (backslash)
	matches <- gregexpr(pattern, args0[4])
	matched_strings <- regmatches(args0[4], matches)

	# Storing index and last index value
	index <- length(matches[[1]])
	last_index <- matches[[1]][index]

	#args0[4]
	#matches[[1]]

	# Get working directory from
	# the command line parameter
	# ---
	# Example: ..\.Rterm -f "..\..\ToPMML.R"
	working_directory <- substr(args0[4], start = matches[[1]], stop=last_index)
	#working_directory	
	print(working_directory)
} else {
	# Substring to search for
	pattern <- "\\\\" #windows character

	# Looking for last occurrence of "\" (backslash)
	matches <- gregexpr(pattern, args0[3])
	matched_strings <- regmatches(args0[3], matches)

	# Storing index and last index value
	index <- length(matches[[1]])
	last_index <- matches[[1]][index]

	#args0[3]
	#matches[[1]]

	# Get working directory from
	# the command line parameter
	# ---
	# Example: ..\.Rterm -f "..\..\ToPMML.R"
	working_directory <- substr(args0[3], start = matches[[1]], stop=last_index)
	#working_directory
}

#q()

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
# ...
# 781         781   59  13    -8.04    164.82        NA        NA    95.50   28.44    28.51
# 782         782   59  14    -8.04    164.81        NA        NA    93.40   28.67    28.61

#
#

# Load model into variable
model <- rpart(s_s_temp ~ ., data = data)

#
#

# Converting to PMML File
r2pmml(model, paste(working_directory, "Elnino.pmml", sep=""))

#
#

# Message result
print("")
print("")
print("")

print("Process finished successfully.")
print("CSV converted to PMML format.")

print("")
print("")

#
#

