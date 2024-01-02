library(shiny)
library(shinythemes)
library(keras)

# Importing the model for prediction
model <- load_model_tf("KoustubR/GemstoneImagePrediction/Model")
class_name <- readRDS('classname.rds')
defaultpath <- 'KoustubR/GemstoneImagePrediction/dataset-cover.jpg'


############################## UI for Upload ########################################

ui <- fluidPage(theme = shinytheme("united"),
                headerPanel("Gemstone Image Prediction"),
                sidebarPanel( 
                  HTML("<h3>Upload file for prediction</h3>"),
                  fileInput("myFile", "Choose a file", 
                            multiple = FALSE,
                            accept = c('image/png', 'image/jpeg')),
                  actionButton("submitbutton", "Submit", class = "btn btn-primary"),
                ),
                mainPanel(
                  tags$label(h3('Status/Output')),
                  verbatimTextOutput('contents'),
                  tableOutput("tabledata"),
                  imageOutput("Inputimage")
                ), 
)
########################## Server Logic ############################################ 

server <- function(input, output, session) {
  imageInput <- reactive({
    output_folder <- "KoustubR/GemstoneImagePrediction/server_uploads"
    filename <- input$myFile$name
    save_path <- file.path(output_folder, filename)
    file.copy(input$myFile$datapath, save_path, overwrite = TRUE)
    ip <- data.frame(filepaths = save_path)
    ############################### Calling the saved model for prediction#############
    pred_gen <- flow_images_from_dataframe(
      dataframe = ip,
      x_col = 'filepaths',
      y_col = NULL,  # There are no labels for prediction
      target_size = c(224, 224),
      class_mode = NULL,  # No class mode for prediction
      color_mode = 'rgb',
      shuffle = FALSE,  # Set to FALSE for prediction
      batch_size = 1
    )
    
    predictions <- predict_generator(model, pred_gen, steps = 1)
    predicted_classes <- apply(predictions, 1, which.max)
    predicted_probabilities <- apply(predictions, 1, max) * 100
    op <- data.frame(Prediction = class_name[predicted_classes],Prediction_Accuracy = predicted_probabilities)
    print(op)
    return(list(save_path,op))
  })
  
  output$contents <- renderPrint({
    if (input$submitbutton > 0) {
      "Calculation complete."
    } else {
      "Server is ready for prediction."
    }
  })
  ##################################### Rendering the predicted output##################
  output$tabledata <- renderTable({
    if (input$submitbutton > 0) {
      isolate(imageInput()[2])
    }
  })
  
  output$Inputimage <- renderImage({
    if (input$submitbutton > 0) {
      list(src = imageInput()[[1]]) 
    }
    else {
      list(src = defaultpath)
    }
  })
}

###################################
shinyApp(ui, server)