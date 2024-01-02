library(dplyr)
install.packages("caret")
library(caret)
install.packages("EBImage")
install.packages("BiocManager") 
BiocManager::install("EBImage")
library(data.table)
library(reticulate)
library(EBImage)
library(keras)
library(tidyverse)

################################# Preprocessing ###################################

#TRAINING_DIR <- Enter your Training Directory      
#TEST_DIR <- Enter your Test Directory

set.seed(123)
train_preprocess <- function(train_dir, trsplit) {
  sdir <- train_dir
  filepaths <- c()
  labels <- c()
    
  classlist <- list.files(path = sdir, full.names = TRUE)
    
  for (klass in classlist) {
      flist <- list.files(path = klass, full.names = TRUE)
      filepaths <- c(filepaths, flist)
      labels <- c(labels, rep(basename(klass), length(flist)))
    }
    
    filepaths <- trimws(filepaths)
    labels <- trimws(labels)
    
    df <- data.frame(filepaths = filepaths, labels = labels)
    
    train_df <- df
    
    strat <- df$labels
    set.seed(123)
    split_indices <- createDataPartition(strat, p = trsplit, list = FALSE)
    train_df <- df[split_indices, ]
    valid_df <- df[-split_indices, ]
    
  
  
  cat('train_df length:', nrow(train_df), 'valid_df length:', nrow(valid_df))
  print(table(train_df$labels))
  
  return(list(train_df = train_df, valid_df = valid_df))
}


test_preprocess <- function(test_dir) {
  sdir <- test_dir
  filepaths <- c()
  labels <- c()
  
  classlist <- list.files(path = sdir, full.names = TRUE)
  
  for (klass in classlist) {
    flist <- list.files(path = klass, full.names = TRUE)
    filepaths <- c(filepaths, flist)
    labels <- c(labels, rep(basename(klass), length(flist)))
  }
  
  filepaths <- trimws(filepaths)
  labels <- trimws(labels)
  
  df <- data.frame(filepaths = filepaths, labels = labels)
  
  test_df <- df
  
  cat('test_df length:', nrow(test_df))
  print(table(test_df$labels))
  
  return(test_df)
}



train_dir <- TRAINING_DIR
test_dir <- TEST_DIR
result <- train_preprocess(train_dir, 0.8)
test_df <- test_preprocess(test_dir)

train_df <- result[[1]]
valid_df <- result[[2]]


head(train_df, 5)
head(test_df, 5)
head(valid_df, 5)


set.seed(123)

##################### Image Augmentation #######################################

img_augmentation <- function(train_df, max_samples, column, AUG_DIR, image_size) {
  train_df <- copy(train_df)
  aug_dir <- AUG_DIR
  
  if (dir.exists(aug_dir)) {
    unlink(aug_dir, recursive = TRUE)
  }
  dir.create(aug_dir)
  
  total <- 0
  gen <- image_data_generator(
    horizontal_flip = TRUE,
    vertical_flip = TRUE,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
    brightness_range = c(0.5, 0.9)
  )
  
  groups <- split(train_df, train_df$labels)
  
  for (label in unique(train_df$labels)) {
    group <- groups[[as.character(label)]]
    sample_count <- nrow(group)
    
    if (sample_count < max_samples) {
      delta <- max_samples - sample_count
      
      target_dir <- file.path(aug_dir, label)
      
      if (any(!sapply(target_dir, dir.exists))) dir.create(target_dir)
      
      aug_gen <- flow_images_from_dataframe(
        dataframe = group,
        directory = '',
        generator = gen,
        x_col = "filepaths",
        y_col = NULL,
        target_size = image_size,
        class_mode = NULL,
        batch_size = 1,
        shuffle = FALSE,
        save_to_dir = target_dir,
        save_prefix = "aug-",
        color_mode = "rgb",
        save_format = "jpg"
      )
      
      for (i in 1:delta) {
        batch <- generator_next(aug_gen)
      }
      
      total <- total + delta
    }
  }
  
  cat("Total Augmented images created = ", total, "\n")
  
  if (total > 0) {
    aug_fpaths <- character()
    aug_labels <- character()
    classlist <- list.dirs(aug_dir, full.names = FALSE, recursive = FALSE)
    
    for (klass in classlist) {
      classpath <- file.path(aug_dir, klass)
      flist <- list.files(classpath, full.names = TRUE)
      
      for (f in flist) {
        aug_fpaths <- c(aug_fpaths, f)
        aug_labels <- c(aug_labels, klass)
      }
    }
    
    aug_df <- data.frame(filepaths = aug_fpaths, labels = aug_labels)
    ndf <- rbindlist(list(train_df, aug_df), use.names = TRUE, fill = TRUE)
  } else {
    ndf <- train_df
  }
  
  print(table(ndf$labels))
  
  return(ndf)
}


#AUG_DIR <- Enter a path to store augmented paths
max_samples <- 150
channels <- 3
img_size <- c(224, 224)

ndf <- img_augmentation(train_df, max_samples, column, AUG_DIR, img_size)
head(ndf, 5)

print(nrow(ndf))

epochs <- 15
batch_size <- 32

train_gen <- flow_images_from_dataframe( dataframe =ndf, 
                                         x_col='filepaths', 
                                         y_col='labels', 
                                         target_size=c(224, 224), 
                                         class_mode='categorical',
                                         color_mode='rgb', 
                                         shuffle=TRUE,
                                         batch_size=batch_size)

print(nrow(ndf))
print(nrow(valid_df))

valid_gen <- flow_images_from_dataframe( dataframe =valid_df, 
                                         x_col='filepaths', 
                                         y_col='labels', 
                                         target_size=c(224, 224), 
                                         class_mode='categorical',
                                         color_mode='rgb', 
                                         shuffle=TRUE)


classes <- names(train_gen$class_indices)
class_count <- length(classes)
print(class_count)

################################### Model Creation #########################################

base_model <- application_efficientnet_b4(
  include_top = FALSE,
  weights = 'imagenet',
  input_shape = c(224, 224, 3),
  pooling = 'max'
)

x <- base_model$output

x %>% layer_batch_normalization(axis = -1, momentum = 0.99, epsilon = 0.001) %>%
  layer_dense(units = 256, 
              kernel_regularizer = regularizer_l2(l = 0.016),
              activity_regularizer = regularizer_l1(l = 0.006),
              bias_regularizer = regularizer_l1(0.006),
              activation = 'relu'
              ) %>%
  layer_dropout(rate = 0.45, seed = 123)

  output <- x %>% layer_dense(units = class_count, activation = 'softmax')


  model <- keras_model(
  inputs = base_model$input,
  outputs = output
)


  model %>% compile(
  optimizer = optimizer_adamax(learning_rate = 0.001),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

train_steps <- ceiling(nrow(ndf) / batch_size)
print(train_steps)
print(epochs)


set.seed(123)
history <- model %>% fit_generator(
  generator = train_gen,
  steps_per_epoch = train_steps,
  epochs = epochs,
  validation_data = valid_gen,
)


test_gen <- flow_images_from_dataframe( dataframe =test_df, 
                                         x_col='filepaths', 
                                         y_col='labels', 
                                         target_size=c(224, 224), 
                                         class_mode='categorical',
                                         color_mode='rgb', 
                                         shuffle=TRUE,
                                         batch_size=33)

model %>% evaluate(test_gen, length(test_gen))

################################ Saving the model and class names for UI ################

save_model_tf(model, "KoustubR/GemstoneImagePrediction/Model")
saveRDS(class_names, "classname.rds")




