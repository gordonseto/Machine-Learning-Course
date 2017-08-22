# Data Preprocessing Template

# Import the dataset
dataset = read.csv('Data.csv')

# Take care of missing data
dataset$Age = ifelse(test = is.na(dataset$Age), 
                     yes = ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     no = dataset$Age)

dataset$Salary = ifelse(test = is.na(dataset$Salary),
                        yes = ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        no = dataset$Salary)

# Encode categorical data
dataset$Country = factor(x = dataset$Country, 
                         levels = c('France', 'Spain', 'Germany'), 
                         labels = c(1, 2, 3))

dataset$Purchased = factor(x = dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))