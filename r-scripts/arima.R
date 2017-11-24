output_dir = '/home/skokov/project/experiments/arma'


ea_data = read.csv('/home/skokov/project/data/EA.csv', row.names = 1, check.names = FALSE)
us_data = read.csv('/home/skokov/project/data/US.csv', row.names = 1, check.names = FALSE)
data = list(EA=ea_data, US=us_data)

  
  
  
  # Multiple plot function
  #
  # ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
  # - cols:   Number of columns in layout
  # - layout: A matrix specifying the layout. If present, 'cols' is ignored.
  #
  # If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
  # then plot 1 will go in the upper left, 2 will go in the upper right, and
  # 3 will go all the way across the bottom.
  #
  multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
    library(grid)
    
    # Make a list from the ... arguments and plotlist
    plots <- c(list(...), plotlist)
    
    numPlots = length(plots)
    
    # If layout is NULL, then use 'cols' to determine layout
    if (is.null(layout)) {
      # Make the panel
      # ncol: Number of columns of plots
      # nrow: Number of rows needed, calculated from # of cols
      layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                       ncol = cols, nrow = ceiling(numPlots/cols))
    }
    
    if (numPlots==1) {
      print(plots[[1]])
      
    } else {
      # Set up the page
      grid.newpage()
      pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
      
      # Make each plot, in the correct location
      for (i in 1:numPlots) {
        # Get the i,j matrix positions of the regions that contain this subplot
        matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
        
        print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                        layout.pos.col = matchidx$col))
      }
    }
  }
  
  
  
  
  
  
  

names(data)

library(forecast)
library(ggplot2)

out= list()

for (country in names(data)){
  for (name in names(data[[country]])){
    print(country)
    print(name)
    
    
    series <- na.omit(ts(data = data[[country]][name], start = c(1998,1), frequency = 4))
    series.train = head(series, -12)
    series.test = tail(series, 12)
    
    
    res = auto.arima(series.train, stepwise = FALSE, approximation = FALSE)
    fcast = forecast(res, h = 12)
    newfit= Arima(series, model = res)
    prediction = fitted(newfit,h = 1)
    
    prediction.train = head(prediction, length(series.train))
    prediction.test = tail(prediction, length(series.test))
    
    
    model = paste('ARIMA(', res$arma[1], ',', res$arma[6], ',', res$arma[2], ')' ,
                  '(', res$arma[3], ',', res$arma[7], ',', res$arma[4], ')',
                  '[', res$arma[5], ']', sep='')
    y_lab=paste(country, name, sep = ' ')
    
    
    
    pdf(file = paste(output_dir, '/', country,'_', name, '_', 'Prediction', '.pdf', sep = ''))
    
    p = autoplot(series, series='observed', ylab=y_lab) + autolayer(prediction.train, series = 'prediction train') + 
      autolayer(prediction.test, series = 'prediction test') + ggtitle(paste('One step prediction from ', model, sep='')) + theme(legend.position="bottom") + 
      theme(text = element_text(size=16))
    print(p)
    
    dev.off()

    pdf(file = paste(output_dir, '/', country,'_', name, '_', 'Forecast', '.pdf', sep = ''))
    
    p = autoplot(fcast, ylab=y_lab) + autolayer(series, series='observed') + autolayer(fcast$mean, series='forecast') +
      ggtitle(paste('Forecasts from ', model, sep = '')) + theme(legend.position="bottom") + 
      theme(text = element_text(size=16))
    print(p)
    
    dev.off()
    
    pdf(file = paste(output_dir, '/', country,'_', name, '_', 'Residuals', '.pdf', sep = ''))
    
    checkresiduals(res, ylab=y_lab, theme=theme(text = element_text(size=16)))
    
    dev.off()

    
    acc.train = c(accuracy(prediction.train, series.train))
    acc.test = accuracy(prediction.test, series.test)
    acc.fcast = accuracy(fcast, series.test)
    
    out=matrix(nrow = 3, ncol = 5)
    rownames(out) <- c('Train', 'Test', 'Forecast')
    colnames(out) <- c('MSE', 'RMSE', 'MAE', 'ME', 'ACF1')
    
    out['Train', 'MSE'] = mean((series.train - prediction.train)^2)
    out['Train', 'RMSE'] = acc.train[2]
    out['Train', 'MAE'] = acc.train[3]
    out['Train', 'ME'] = acc.train[1]
    out['Train', 'ACF1'] = acc.train[6]
      
    out['Test', 'MSE'] = mean((series.test - prediction.test)^2)
    out['Test', 'RMSE'] = acc.test[2]
    out['Test', 'MAE'] = acc.test[3]
    out['Test', 'ME'] = acc.test[1]
    out['Test', 'ACF1'] = acc.test[6]  
    
    out['Forecast', 'MSE'] = mean((series.test - fcast$mean)^2)
    out['Forecast', 'RMSE'] = acc.fcast[2,2]
    out['Forecast', 'MAE'] = acc.fcast[2,3]
    out['Forecast', 'ME'] = acc.fcast[2,1]
    out['Forecast', 'ACF1'] = acc.fcast[2,7]
    
    out <- round(out, 5)
    output = cbind(rownames(out), out)
    write.csv(x = output, file = paste(output_dir, '/',country,'_', name, '_', 'Accuracy', '.csv', sep = ''), quote = FALSE, row.names = FALSE)
    
    summary(res)
    print(out)
  }
}

