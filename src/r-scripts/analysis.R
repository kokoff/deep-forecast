library(readr)
library(tseries)
library(urca)
library(forecast)
library(ggplot2)

WORK_DIR <- '/home/skokov/project/experiments/r-analysis'
DATA_DIR <- '/home/skokov/project/data/'
DATA_FILES <- list('EA.csv', 'US.csv')
DATA_NAMES <- list('EA', 'US')


#------------------------------------------------------------------------------------
# Read the data into list of lists of ts
#------------------------------------------------------------------------------------

data <- list(list(), list())
names(data) <- DATA_NAMES

for (i in 1:length(DATA_FILES)){
    path <- paste(DATA_DIR, DATA_FILES[i], sep = '/')
    temp <- read_csv(path)
    print(temp)
    
    for (j in 2:length(temp)){
      series <- ts(data = temp[j], start = c(1998,1), frequency = 4)
      data_name <- DATA_NAMES[[i]]
      
      var_name <- names(temp)[j]
      print (var_name)
      data[[data_name]][[var_name]] <- series
    }
}



#------------------------------------------------------------------------------------
# Plot the data
#------------------------------------------------------------------------------------

for (i in names(data)){
  for (j in names(data[[i]])){
    path <- paste(WORK_DIR,'/', i, '_', j, '_','Plot', ".pdf", sep="")
    pdf(file=path)
    series <- data[[i]][[j]] 
    print(autoplot(series, ylab=paste(i,j,sep = ' ')))
    dev.off()
  }
}


#------------------------------------------------------------------------------------
# Plot the ACFs
#------------------------------------------------------------------------------------

for (i in names(data)){
  for (j in names(data[[i]])){
    path <- paste(WORK_DIR,'/', i, '_', j, '_', 'ACF_plot', ".pdf", sep="")
    pdf(file=path)
    series <- data[[i]][[j]] 
    print(ggAcf(series, main=paste(i,j,sep = ' ')))
    dev.off()
  }
}


#------------------------------------------------------------------------------------
# Plot the PACFs
#------------------------------------------------------------------------------------

for (i in names(data)){
  for (j in names(data[[i]])){
    path <- paste(WORK_DIR,'/', '/', i, '_', j, '_', 'PACF_plot', ".pdf", sep="")
    pdf(file=path)
    series <- data[[i]][[j]] 
    print(ggPacf(series, main=paste(i,j,sep = ' ')))
    dev.off()
  }
}


#------------------------------------------------------------------------------------
# Run Stationarity Tests
#------------------------------------------------------------------------------------


round_df <- function(x, digits) {
  # round all numeric variables
  # x: data frame 
  # digits: number of digits to round
  numeric_columns <- sapply(x, mode) == 'numeric'
  x[numeric_columns] <-  round(x[numeric_columns], digits)
  x
}


filepath <- paste(WORK_DIR, 'tests_results.txt', sep = '/')
sink(filepath)

for (i in names(data)){
  for (j in names(data[[i]])){
    filename <- paste(i, '_',  j, '_', 'stationary_tests', '.csv', sep = '')
    filepath <- paste(WORK_DIR, filename, sep = '/')
    
    series <- data[[i]][[j]]
    series <- na.omit(series)
    
    res.df1 = ur.df(series, type = 'none', selectlags = 'AIC',)
    res.df2 = ur.df(series, type = 'drift', selectlags = 'AIC')
    res.df3 = ur.df(series, type = 'trend', selectlags = 'AIC')
    res.pp1 = ur.pp(series, model = 'constant', type = 'Z-tau')
    res.pp2 = ur.pp(series, model = 'trend', type = 'Z-tau')
    res.kpss1 = ur.kpss(series, type = 'mu')
    res.kpss2 = ur.kpss(series, type = 'tau')
    
    a <- data.frame('ADF', res.df1@teststat,  res.df1@cval[1], res.df1@cval[2], res.df1@cval[3], ifelse(res.df1@teststat[1] < res.df1@cval[2], 'Stationary', 'Not Stationary'), stringsAsFactors=FALSE)
    a <- rbind(a, list('ADF with drift',  res.df2@teststat[1],  res.df2@cval[1,1], res.df2@cval[1,2], res.df2@cval[1,3], ifelse(res.df2@teststat[1] < res.df2@cval[1,2], 'Stationary', 'Not Stationary') )) 
    a <- rbind(a, list('ADF with drift and trend',  res.df3@teststat[1],  res.df3@cval[1,1], res.df3@cval[1,2], res.df3@cval[1,3], ifelse(res.df3@teststat[1] < res.df3@cval[1,2], 'Stationary', 'Not Stationary') ))
    
    a <- rbind(a, list('PP with drift',  res.pp1@teststat[1],  res.pp1@cval[1], res.pp1@cval[2], res.pp1@cval[3], ifelse(res.pp1@teststat[1] < res.pp1@cval[2], 'Stationary', 'Not Stationary') ))
    a <- rbind(a, list('PP with drift and trend',  res.pp2@teststat[1],  res.pp2@cval[1], res.pp2@cval[2], res.pp2@cval[3], ifelse(res.pp2@teststat[1] < res.pp2@cval[2], 'Stationary', 'Not Stationary') ))
    
    a <- rbind(a, list('KPSS with drift',  res.kpss1@teststat[1],  res.kpss1@cval[4], res.kpss1@cval[2], res.kpss1@cval[1], ifelse(res.kpss1@teststat[1] < res.kpss1@cval[2], 'Not Stationary', 'Stationary')))
    a <- rbind(a, list('KPSS with drift and trend',  res.kpss2@teststat[1],  res.kpss2@cval[4], res.kpss2@cval[2], res.kpss2@cval[1], ifelse(res.kpss2@teststat[1] < res.kpss2@cval[2], 'Not Stationary', 'Stationary')))
    
    rownames(a) <- c()
    colnames(a) <- c('Test Name', 'T-stat', '1%', '5%', '10%', '>95% Confidence')
  
    a <- round_df(a, 3)  
    write.csv(x = a,file = filepath, quote = FALSE, row.names = FALSE)
    
    
    print(paste(rep('#', 100), collapse = ''))
    print(i)
    print(j)
    print(paste(rep('#', 100), collapse = ''))
    
    print(summary(res.df1))
    print(summary(res.df2))
    print(summary(res.df3))
    print(summary(res.pp1))
    print(summary(res.pp2))
    print(summary(res.kpss1))
    print(summary(res.kpss2))
    
  }
}

closeAllConnections()


#------------------------------------------------------------------------------------
# Run Stationarity Tests on Differenced data
#------------------------------------------------------------------------------------

for (i in names(data)){
  for (j in names(data[[i]])){
    filename <- paste('diff', '_', i, '_',  j, '.csv', sep = '')
    filepath <- paste(WORK_DIR, 'stationary_tests', filename, sep = '/')
    
    series <- data[[i]][[j]]
    series <- na.omit(series)
    series <- diff(series)
    #series <- log(series - min(series) + 10)
    
    
    res.df1 = ur.df(series, type = 'none', selectlags = 'AIC',)
    res.df2 = ur.df(series, type = 'drift', selectlags = 'AIC')
    res.df3 = ur.df(series, type = 'trend', selectlags = 'AIC')
    res.pp1 = ur.pp(series, model = 'constant', type = 'Z-tau')
    res.pp2 = ur.pp(series, model = 'trend', type = 'Z-tau')
    res.kpss1 = ur.kpss(series, type = 'mu')
    res.kpss2 = ur.kpss(series, type = 'tau')
    
    a <- data.frame('ADF', res.df1@teststat,  res.df1@cval[1], res.df1@cval[2], res.df1@cval[3], ifelse(res.df1@teststat[1] < res.df1@cval[2], 'Stationary', 'Not Stationary'), stringsAsFactors=FALSE)
    a <- rbind(a, list('ADF with drift',  res.df2@teststat[1],  res.df2@cval[1,1], res.df2@cval[1,2], res.df2@cval[1,3], ifelse(res.df2@teststat[1] < res.df2@cval[1,2], 'Stationary', 'Not Stationary') )) 
    a <- rbind(a, list('ADF with drift and trend',  res.df3@teststat[1],  res.df3@cval[1,1], res.df3@cval[1,2], res.df3@cval[1,3], ifelse(res.df3@teststat[1] < res.df3@cval[1,2], 'Stationary', 'Not Stationary') ))
    
    a <- rbind(a, list('PP with drift',  res.pp1@teststat[1],  res.pp1@cval[1], res.pp1@cval[2], res.pp1@cval[3], ifelse(res.pp1@teststat[1] < res.pp1@cval[2], 'Stationary', 'Not Stationary') ))
    a <- rbind(a, list('PP with drift and trend',  res.pp2@teststat[1],  res.pp2@cval[1], res.pp2@cval[2], res.pp2@cval[3], ifelse(res.pp2@teststat[1] < res.pp2@cval[2], 'Stationary', 'Not Stationary') ))
    
    a <- rbind(a, list('KPSS with drift',  res.kpss1@teststat[1],  res.kpss1@cval[4], res.kpss1@cval[2], res.kpss1@cval[1], ifelse(res.kpss1@teststat[1] < res.kpss1@cval[2], 'Not Stationary', 'Stationary')))
    a <- rbind(a, list('KPSS with drift and trend',  res.kpss2@teststat[1],  res.kpss2@cval[4], res.kpss2@cval[2], res.kpss2@cval[1], ifelse(res.kpss2@teststat[1] < res.kpss2@cval[2], 'Not Stationary', 'Stationary')))
    
    rownames(a) <- c()
    colnames(a) <- c('Test Name', 'T-stat', '1%', '5%', '10%', '>95% Confidence')
    
    a <- round_df(a, 3)  
    write.csv(x = a,file = filepath, quote = FALSE, row.names = FALSE)
    print(j)
    print(a)
    
  }
}
  
