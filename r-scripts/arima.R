output_dir = '/home/skokov/project/experiments/arma'


ea_data = read.csv('/home/skokov/project/data/EA.csv', row.names = 1)
us_data = read.csv('/home/skokov/project/data/US.csv', row.names = 1)
data = list(EA=ea_data, US=us_data)

names(data)

library(forecast)

out= list()

for (country in names(data)){
  for (name in names(data[[country]])){
    print(country)
    print(name)
    

    
    series <- na.omit(ts(data = data[[country]][name], start = c(1998,1), frequency = 4))
    train = head(series, -12)
    test = tail(series, 12)
    
    
    res = auto.arima(train, stepwise = FALSE, approximation = FALSE)
    fcast = forecast(res, h = 12)
    newfit= Arima(series, model = res)
    fit = fitted(newfit,h = 1)
    
    
    
    
    png(filename = paste(output_dir, '/', country,'_', name,'.png', sep = ''), width=700, height = 1000)
    par(mfrow=c(3,2))
    
    plot(series, col='red')
    lines(fit, col='blue', lwd=2)
    legend('topleft', legend = c('original', 'one step predict'), col=c('blue', 'red'), lty=1)
    
    plot(fcast, col=c('blue', 'blue', 'red'))
    lines(series, col='red', lwd=1)
    
    legend('topleft', legend = c('recursive forecast', 'original'), col=c('blue', 'red'), lty=1)
    
    plot(residuals(res))
    plot(density(residuals(res)))
    Acf(residuals(res))
    Pacf(residuals(res))
    dev.off()
    
    out = as.list(res$arma)
    names(out) = c('p', 'd','q', 'P', 'D', 'Q', 'asd')
    
    out['Train MSE'] = mean((head(fit, length(train)) - train)^2)
    out['Test MSE'] = mean((tail(fit, length(test)) - test)^2)
    out['Test MSE forecast'] = mean((fcast$mean - test)^2)
    
    out[] <- lapply(out,round,5)
    write.csv(x = out,file = paste(output_dir, '/',country,'_', name,'.csv', sep = ''), quote = FALSE, row.names = FALSE)
    
    summary(res)
    print('MSE Test:')
    print(mean((fcast$mean - test)^2))
  }
}
