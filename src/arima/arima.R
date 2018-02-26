library(forecast)
library(ggplot2)
library(Metrics)
library(zoo)

#output_dir = '/home/skokov/project/experiments/arima'


ea_data = read.csv('/home/skokov/project/data/EA.csv', row.names = 1, check.names = FALSE)
us_data = read.csv('/home/skokov/project/data/US.csv', row.names = 1, check.names = FALSE)
data = list(EA = ea_data, US = us_data)


names(data)



out = list()

data$EA = na.omit(data$EA) #remove na rows
data$US = na.omit(data$US) # remove na rows

for (country in names(data)) {
  for (name in names(data[[country]])) {
    print(country)
    print(name)
    
    
    series <- na.omit(ts(data = data[[country]][name], start = c(1999, 1), frequency = 4))
    series.train = head(series, - 24)
    series.train_val = head(series, -12)
    series.val = head(tail(series, 24), 12)
    series.test = tail(series, 12)
    
    
    model_fit = auto.arima(series.train, stepwise = FALSE, approximation = FALSE)
    val_fit = Arima(series.train_val, model = model_fit)
    test_fit = Arima(series, model = model_fit)
    
    # evaluate predictions
    prediction.true = series
    prediction.train = fitted(model_fit)
    prediction.val = tail(fitted(val_fit), 12)
    prediction.test = tail(fitted(test_fit), 12)
    prediction = cbind(prediction.true, prediction.train, prediction.val, prediction.test)
    #prediction = ts(data=prediction, start = c(1999,2), frequency = 4)
    #prediction = tail(head(cbind(prediction.true, prediction), -1), -1)
    
    
    # evaluate forecasts
    forecast.true = series
    forecast.train= fitted(model_fit)
    forecast.val = tail(forecast(model_fit, h=12)$mean, 12)
    forecast.test = tail(forecast(val_fit, h=12)$mean, 12)
    forecast = cbind(forecast.true, forecast.train, forecast.val, forecast.test)
    #forecast = ts(data=forecast, start = c(1999,2), frequency = 4)
    #forecast = tail(head(cbind(forecast.true, forecast), -1), -1)
    
    # calculate mse
    out = matrix(nrow = 1, ncol = 6)
    rownames(out) = c('MSE')
    colnames(out) = c("train pred","train fcast","val pred","val fcast","test pred","test fcast")
    
    out['MSE', 'train pred'] = mse(series.train, prediction.train)
    out['MSE', 'train fcast'] = mse(series.train, forecast.train)
    out['MSE', 'val pred'] = mse(series.val, prediction.val)
    out['MSE', 'val fcast'] = mse(series.val, forecast.val)
    out['MSE', 'test pred'] = mse(series.test, prediction.test)
    out['MSE', 'test fcast'] = mse(series.test, forecast.test)
    
    # output files
    dir_path = file.path(output_dir, paste(country, '_', name, sep=''))
    dir.create(dir_path)
    
    
    # out <- round(out, 5)
    output = cbind(rownames(out), out)
    file_name = file.path(dir_path, "performance.csv")
    write.csv(x = output, file = file_name, quote = FALSE, row.names = FALSE)
    
    colnames(prediction) = c("true values", "train prediction", "val prediction", "test prediction")
    file_name = file.path(dir_path, "predictions.csv")
    write.zoo(as.zoo(prediction), file=file_name, quote = FALSE, sep=',')
    
    colnames(forecast) = c("true values", "train forecast", "val forecast", "test forecast")
    file_name = file.path(dir_path, "forecasts.csv")
    write.zoo(as.zoo(forecast), file=file_name, quote = FALSE, sep=',')
    
    file_name = file.path(dir_path, "summary.txt")
    
    sink(file_name)
    print(summary(model_fit))
    sink()  #
    
    summary(model_fit)
    print(out)
  }
}

data = series.train

model_fit = auto.arima(data, stepwise = FALSE, approximation = FALSE)
prediction = fitted(model_fit)

print(data)
print(prediction)

