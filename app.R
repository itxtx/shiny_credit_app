# --- Install necessary packages if you haven't already ---
# install.packages(c("shiny", "shinydashboard", "ggplot2", "dplyr", "DT", "RColorBrewer", "PRROC", "markdown"))

library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(DT)
library(RColorBrewer)
library(PRROC)
library(markdown)

# --- Data Loading (for all models) ---
load_safe_csv <- function(filename) {
  if (file.exists(filename)) {
    df <- read.csv(filename, stringsAsFactors = FALSE)
    if (!("true_label" %in% names(df))) return(NULL)
    if ("predicted_probability" %in% names(df)) {
      df$predicted_score <- df$predicted_probability
    } else if ("anomaly_score" %in% names(df)) {
      df$predicted_score <- df$anomaly_score
    } else {
      return(NULL)
    }
    df$predicted_score <- as.numeric(as.character(df$predicted_score))
    df$true_label <- as.numeric(as.character(df$true_label))
    df <- df[complete.cases(df$predicted_score, df$true_label), ]
    if (nrow(df) == 0) return(NULL)
    return(df)
  } else {
    return(NULL)
  }
}

# --- Load all prediction files ---
xgb_preds <- load_safe_csv("xgb_predictions.csv")
lr_preds <- load_safe_csv("lr_predictions.csv")
rf_preds <- load_safe_csv("rf_predictions.csv")
if_preds <- load_safe_csv("if_predictions.csv")
lof_preds <- load_safe_csv("lof_predictions.csv")
ae_preds <- load_safe_csv("ae_predictions.csv")

# --- Helper function for metrics ---
calculate_metrics <- function(y_true, y_pred_scores, threshold, direction = "gte") {
  if (length(y_true) == 0) return(list(precision = 0, recall = 0, f1_score = 0, mcc = 0, fpr_at_threshold = 0, tpr_at_threshold = 0, specificity = 0))
  
  y_pred_binary <- if (direction == "gte") {
    as.numeric(y_pred_scores >= threshold)
  } else {
    as.numeric(y_pred_scores <= threshold)
  }
  
  tp <- sum(y_true == 1 & y_pred_binary == 1)
  fp <- sum(y_true == 0 & y_pred_binary == 1)
  tn <- sum(y_true == 0 & y_pred_binary == 0)
  fn <- sum(y_true == 1 & y_pred_binary == 0)
  
  precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  recall <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  f1_score <- ifelse(precision + recall == 0, 0, 2 * (precision * recall) / (precision + recall))
  numerator_mcc <- as.double(tp) * as.double(tn) - as.double(fp) * as.double(fn)
  denominator_mcc <- sqrt(as.double(tp + fp) * as.double(tp + fn) * as.double(tn + fp) * as.double(tn + fn))
  mcc <- ifelse(denominator_mcc == 0, 0, numerator_mcc / denominator_mcc)
  fpr_at_threshold <- ifelse(fp + tn == 0, 0, fp / (fp + tn))
  specificity <- ifelse(tn + fp == 0, 0, tn / (tn + fp))  # True Negative Rate
  
  return(list(
    confusion_matrix = matrix(c(tn, fp, fn, tp), nrow = 2, byrow = TRUE, dimnames = list(c("Actual 0", "Actual 1"), c("Pred 0", "Pred 1"))),
    precision = precision, recall = recall, f1_score = f1_score, mcc = mcc,
    fpr_at_threshold = fpr_at_threshold, tpr_at_threshold = recall, specificity = specificity
  ))
}

# --- UI (User Interface) ---
ui <- dashboardPage(
  dashboardHeader(title = "Fraud Detection Dashboard"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dashboard", tabName = "dashboard", icon = icon("dashboard")),
      menuItem("Model Comparison", tabName = "comparison", icon = icon("chart-line")),
      menuItem("Model Overview", tabName = "overview", icon = icon("info-circle"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(
        tabName = "dashboard",
        fluidRow(
          box(
            title = "Controls", status = "primary", solidHeader = TRUE, width = 4,
            selectInput(
              inputId = "model_select",
              label = "Select Model:",
              choices = {
                model_choices <- c()
                if (!is.null(xgb_preds)) model_choices["XGBoost"] <- "xgb"
                if (!is.null(rf_preds)) model_choices["Random Forest"] <- "rf"
                if (!is.null(lr_preds)) model_choices["Logistic Regression"] <- "lr"
                if (!is.null(if_preds)) model_choices["Isolation Forest"] <- "if"
                if (!is.null(lof_preds)) model_choices["Local Outlier Factor"] <- "lof"
                if (!is.null(ae_preds)) model_choices["Autoencoder"] <- "ae"
                if (length(model_choices) == 0) model_choices["Demo Model"] <- "demo"
                model_choices
              }
            ),
            uiOutput("dynamic_threshold_slider")
          ),
          box(
            title = "Confusion Matrix", status = "warning", solidHeader = TRUE, width = 4,
            tableOutput("confusion_matrix")
          ),
          box(
            title = "Performance Metrics", status = "success", solidHeader = TRUE, width = 4,
            uiOutput("metrics_warning"),
            h4("Precision: ", textOutput("precision", inline = TRUE)),
            h4("Recall (TPR): ", textOutput("recall", inline = TRUE)),
            h4("F1-Score: ", textOutput("f1_score", inline = TRUE)),
            h4("MCC: ", textOutput("mcc", inline = TRUE)),
            h4("Specificity: ", textOutput("specificity", inline = TRUE))
          )
        ),
        fluidRow(
          conditionalPanel(
            condition = "input.model_select != 'if' && input.model_select != 'lof' && input.model_select != 'ae'",
            box(
              title = "Precision-Recall Curve", status = "info", solidHeader = TRUE, width = 6,
              plotOutput("pr_curve_plot")
            )
          ),
          conditionalPanel(
            condition = "input.model_select != 'if' && input.model_select != 'lof' && input.model_select != 'ae'",
            box(
              title = "ROC Curve", status = "info", solidHeader = TRUE, width = 6,
              plotOutput("roc_curve_plot")
            )
          ),
          conditionalPanel(
            condition = "input.model_select == 'if' || input.model_select == 'lof' || input.model_select == 'ae'",
            box(
              title = "Anomaly Score Distribution", status = "warning", solidHeader = TRUE, width = 12,
              plotOutput("anomaly_distribution_plot")
            )
          )
        ),
        fluidRow(
          box(
            title = "Predicted Score Distribution", status = "info", solidHeader = TRUE, width = 8,
            plotOutput("score_distribution_plot")
          ),
          box(
            title = "Summary Statistics", status = "primary", solidHeader = TRUE, width = 4,
            tableOutput("summary_stats")
          )
        ),
        fluidRow(
          conditionalPanel(
            condition = "input.model_select == 'xgb' || input.model_select == 'rf' || input.model_select == 'lr'",
            tabBox(
              title = "Advanced Metrics for Selected Model", width = 12,
              tabPanel("Threshold Analysis", plotOutput("threshold_analysis_plot")),
              tabPanel("Cumulative Gain", plotOutput("cumulative_gain_plot")),
              tabPanel("Class-wise Metrics", plotOutput("class_wise_metrics_plot")),
              tabPanel("Calibration", plotOutput("calibration_plot")),
              tabPanel("KS Statistic", plotOutput("ks_plot")),
              tabPanel("Score Distribution by Class", plotOutput("score_distribution_by_class_plot"))
            )
          )
        ),

      ),
      tabItem(
        tabName = "comparison",
        fluidRow(
          box(
            title = "Model Performance Comparison", status = "primary", solidHeader = TRUE, width = 12,
            plotOutput("model_comparison_plot")
          )
        ),
        fluidRow(
          box(
            title = "Model Metrics Table", status = "info", solidHeader = TRUE, width = 12,
            tableOutput("model_comparison_table")
          )
        )
      ),

      tabItem(
        tabName = "overview",
        uiOutput("readme_content")
      )
    )
  )
)


# --- Server Logic ---
server <- function(input, output, session) {
  
  # --- Configuration Maps ---
  
  # Defines if a higher score ("gte") or lower score ("lte") indicates fraud.
  model_score_directions <- c(
    "xgb" = "gte",
    "rf" = "gte",
    "lr" = "gte",
    "ae" = "gte",
    "if" = "lte",
    "lof" = "lte",
    "demo" = "gte"
  )
  
  # **NEW:** Optimal F1-score thresholds to be used as defaults.
  optimal_f1_thresholds <- c(
    "rf" = 0.490,
    "xgb" = 0.740,
    "lr" = 0.190,
    "if" = 0.500,  # Default for Isolation Forest
    "lof" = 0.500, # Default for Local Outlier Factor
    "ae" = 0.500   # Default for Autoencoder
  )
  
  # --- Reactive Data Handling ---
  
  create_demo_data <- function() {
    set.seed(42)
    n <- 2000
    labels <- rbinom(n, 1, 0.05)
    scores <- ifelse(labels == 1, rbeta(n, 7, 3), rbeta(n, 2, 5))
    data.frame(true_label = labels, predicted_score = scores)
  }
  
  selected_data <- reactive({
    req(input$model_select)
    data <- switch(
      input$model_select,
      "xgb" = xgb_preds, "rf" = rf_preds, "lr" = lr_preds,
      "if" = if_preds, "lof" = lof_preds, "ae" = ae_preds,
      "demo" = create_demo_data()
    )
    if (is.null(data)) {
      demo_data <- create_demo_data()
      return(demo_data)
    } else {
      # Ensure data has required columns
      if (!all(c("true_label", "predicted_score") %in% names(data))) {
        return(create_demo_data())
      }
      return(data)
    }
  })
  
  output$dynamic_threshold_slider <- renderUI({
    req(input$model_select)
    scores <- selected_data()$predicted_score
    
    # **MODIFIED:** Set initial value from optimal thresholds map, or fall back to median.
    initial_val <- optimal_f1_thresholds[[input$model_select]]
    if (is.null(initial_val)) {
      initial_val <- quantile(scores, 0.5, na.rm = TRUE)
    }
    
    min_val <- min(scores, na.rm = TRUE)
    max_val <- max(scores, na.rm = TRUE)
    slider_step <- round((max_val - min_val) / 200, 5)
    
    # Ensure we have valid values
    if (is.na(min_val) || is.na(max_val) || is.na(initial_val)) {
      min_val <- 0
      max_val <- 1
      initial_val <- 0.5
    }
    
    sliderInput(
      "threshold", 
      if (input$model_select %in% c("if", "lof", "ae")) "Anomaly Threshold:" else "Classification Threshold:",
      min = round(min_val, 4), max = round(max_val, 4), 
      value = round(initial_val, 4), 
      step = if(slider_step > 0) slider_step else 0.001
    )
  })
  
  metrics_output <- reactive({
    req(input$threshold, input$model_select)
    data <- selected_data()
    direction <- model_score_directions[[input$model_select]]
    calculate_metrics(data$true_label, data$predicted_score, input$threshold, direction)
  })
  
  output$confusion_matrix <- renderTable({ metrics_output()$confusion_matrix }, rownames = TRUE)
  output$precision <- renderText({ round(metrics_output()$precision, 4) })
  output$recall <- renderText({ round(metrics_output()$recall, 4) })
  output$f1_score <- renderText({ round(metrics_output()$f1_score, 4) })
  output$mcc <- renderText({ round(metrics_output()$mcc, 4) })
  output$specificity <- renderText({ round(metrics_output()$specificity, 4) })
  

  
  curve_data_obj <- reactive({
    req(input$model_select)
    data <- selected_data()
    
    # Get the direction for this model
    direction <- model_score_directions[[input$model_select]]
    
    # For ROC curves, we still use PRROC
    scores <- data$predicted_score
    if (direction == "lte") {
      # Invert scores so higher values indicate fraud for PRROC
      scores <- -scores
    }
    
    scores_class0 <- scores[data$true_label == 0]
    scores_class1 <- scores[data$true_label == 1]
    
    if (length(scores_class0) > 0 && length(scores_class1) > 0) {
      roc_result <- roc.curve(scores.class0 = scores_class0, scores.class1 = scores_class1, curve = TRUE)
      
      # Calculate threshold-based PR curve manually
      # Generate thresholds from min to max score
      score_range <- range(data$predicted_score)
      thresholds <- seq(score_range[1], score_range[2], length.out = 100)
      
      pr_points <- data.frame(threshold = thresholds, precision = NA, recall = NA)
      
      for (i in seq_along(thresholds)) {
        thresh <- thresholds[i]
        metrics <- calculate_metrics(data$true_label, data$predicted_score, thresh, direction)
        pr_points$precision[i] <- metrics$precision
        pr_points$recall[i] <- metrics$recall
      }
      
      # Calculate AUCPR using trapezoidal rule
      aucpr <- sum(diff(pr_points$recall) * (pr_points$precision[-1] + pr_points$precision[-length(pr_points$precision)]) / 2)
      
      return(list(pr = list(curve = as.matrix(pr_points[, c("recall", "precision")]), auc.integral = aucpr), 
                  roc = roc_result, 
                  model = input$model_select))
    } else {
      return(NULL)
    }
  })
  
  # --- Plots ---
  
  output$pr_curve_plot <- renderPlot({
    req(input$threshold)
    pr_result <- curve_data_obj()$pr
    
    if (is.null(pr_result) || is.null(pr_result$curve)) {
      return(ggplot() + annotate("text", x=0.5, y=0.5, label="Not enough data for PR curve.") + theme_void())
    }
    
    pr_plot_data <- data.frame(recall = pr_result$curve[, 1], precision = pr_result$curve[, 2])
    metrics_at_threshold <- metrics_output()
    
    # Add debugging info
    data <- selected_data()
    n_fraud <- sum(data$true_label == 1)
    n_total <- nrow(data)
    score_range <- range(data$predicted_score)
    
    p <- ggplot(pr_plot_data, aes(x = recall, y = precision)) +
      geom_line(color = "#E41A1C", linewidth = 1) +
      annotate("point", x = metrics_at_threshold$recall, y = metrics_at_threshold$precision,
               color = "red", size = 4, shape = 19, alpha = 0.8) +
      annotate("text", x = metrics_at_threshold$recall, y = metrics_at_threshold$precision,
               label = paste0("Thresh: ", round(input$threshold, 3)), vjust = -1.5, color = "red") +
      labs(title = paste0(input$model_select, " - Precision-Recall Curve (AUCPR = ", round(pr_result$auc.integral, 4), ")"),
           subtitle = paste("Model:", input$model_select, "| Data:", n_fraud, "fraud cases out of", n_total, "total (", round(100*n_fraud/n_total, 2), "% fraud rate) | Score range:", round(score_range[1], 6), "to", round(score_range[2], 6)),
           x = "Recall", y = "Precision") +
      theme_minimal() + coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))
    return(p)
  })
  
  output$roc_curve_plot <- renderPlot({
    req(input$threshold)
    roc_result <- curve_data_obj()$roc
    
    if (is.null(roc_result) || is.null(roc_result$curve)) {
      return(ggplot() + annotate("text", x=0.5, y=0.5, label="Not enough data for ROC curve.") + theme_void())
    }
    
    roc_plot_data <- data.frame(tpr = roc_result$curve[, 1], fpr = roc_result$curve[, 2])
    metrics_at_threshold <- metrics_output()
    
    # Add debugging info
    data <- selected_data()
    score_range <- range(data$predicted_score)
    
    p <- ggplot(roc_plot_data, aes(x = fpr, y = tpr)) +
      geom_line(color = "#E41A1C", linewidth = 1) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey") +
      annotate("point", x = metrics_at_threshold$fpr_at_threshold, y = metrics_at_threshold$tpr_at_threshold,
               color = "red", size = 4, shape = 19, alpha = 0.8) +
      annotate("text", x = metrics_at_threshold$fpr_at_threshold, y = metrics_at_threshold$tpr_at_threshold,
               label = paste0("Thresh: ", round(input$threshold, 3)), vjust = -1.5, color = "red") +
      labs(title = paste0(input$model_select, " - ROC Curve (AUROC = ", round(roc_result$auc, 4), ")"),
           subtitle = paste("Model:", input$model_select, "| Score range:", round(score_range[1], 6), "to", round(score_range[2], 6)),
           x = "False Positive Rate (FPR)", y = "True Positive Rate (TPR)") +
      theme_minimal() + coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))
    return(p)
  })
  
  output$score_distribution_plot <- renderPlot({
    req(input$threshold)
    data <- selected_data()
    plot_data <- data.frame(
      Score = data$predicted_score,
      Class = factor(data$true_label, levels = c(0, 1), labels = c("Normal", "Fraud"))
    )
    
    # Calculate optimal number of bins based on data size
    n_bins <- min(50, max(20, round(nrow(plot_data) / 100)))
    
    # Create faceted histogram for better visibility
    p <- ggplot(plot_data, aes(x = Score, fill = Class)) +
      geom_histogram(bins = n_bins, alpha = 0.7, position = "identity") +
      geom_vline(xintercept = input$threshold, linetype = "dashed", color = "red", linewidth = 1.2) +
      annotate("text", x = input$threshold, y = Inf,
               label = paste0("Thresh: ", round(input$threshold, 3)),
               vjust = 1.5, hjust = ifelse(input$threshold > median(plot_data$Score), 1.1, -0.1), color = "red") +
      labs(title = "Predicted Score Distribution by Class", 
           subtitle = paste("Histogram with", n_bins, "bins"),
           x = "Predicted Score", y = "Count") +
      scale_fill_manual(values = c("Normal" = "#377EB8", "Fraud" = "#E41A1C")) +
      theme_minimal() + 
      theme(legend.position = "top") +
      facet_wrap(~Class, scales = "free_y", ncol = 1)
    
    return(p)
  })
  
  output$anomaly_distribution_plot <- renderPlot({
    req(input$threshold)
    data <- selected_data()
    
    # For unsupervised models, show anomaly score distribution
    plot_data <- data.frame(
      AnomalyScore = data$predicted_score,
      Class = factor(data$true_label, levels = c(0, 1), labels = c("Normal", "Fraud"))
    )
    
    # Calculate percentiles for context
    p95 <- quantile(plot_data$AnomalyScore, 0.95, na.rm = TRUE)
    p99 <- quantile(plot_data$AnomalyScore, 0.99, na.rm = TRUE)
    
    p <- ggplot(plot_data, aes(x = AnomalyScore)) +
      geom_histogram(bins = 50, alpha = 0.7, fill = "#377EB8") +
      geom_vline(xintercept = input$threshold, linetype = "dashed", color = "red", linewidth = 1.2) +
      geom_vline(xintercept = p95, linetype = "dotted", color = "orange", linewidth = 1) +
      geom_vline(xintercept = p99, linetype = "dotted", color = "darkorange", linewidth = 1) +
      annotate("text", x = input$threshold, y = Inf,
               label = paste0("Threshold: ", round(input$threshold, 3)),
               vjust = 1.5, hjust = ifelse(input$threshold > median(plot_data$AnomalyScore), 1.1, -0.1), color = "red") +
      annotate("text", x = p95, y = Inf,
               label = paste0("95th percentile: ", round(p95, 3)),
               vjust = 3, hjust = -0.1, color = "orange", size = 3) +
      annotate("text", x = p99, y = Inf,
               label = paste0("99th percentile: ", round(p99, 3)),
               vjust = 5, hjust = -0.1, color = "darkorange", size = 3) +
      labs(title = "Anomaly Score Distribution", 
           subtitle = "Red line = threshold, Orange lines = percentiles",
           x = "Anomaly Score", y = "Count") +
      theme_minimal()
    
    return(p)
  })
  
  output$summary_stats <- renderTable({
    req(input$model_select)
    data <- selected_data()
    
    # Calculate summary statistics
    stats <- data.frame(
      Metric = c("Total Samples", "Fraud Cases", "Normal Cases", "Fraud Rate (%)", 
                 "Mean Score", "Median Score", "Std Dev", "Min Score", "Max Score"),
      Value = c(
        nrow(data),
        sum(data$true_label == 1),
        sum(data$true_label == 0),
        round(100 * sum(data$true_label == 1) / nrow(data), 2),
        round(mean(data$predicted_score, na.rm = TRUE), 4),
        round(median(data$predicted_score, na.rm = TRUE), 4),
        round(sd(data$predicted_score, na.rm = TRUE), 4),
        round(min(data$predicted_score, na.rm = TRUE), 4),
        round(max(data$predicted_score, na.rm = TRUE), 4)
      )
    )
    
    return(stats)
  }, rownames = FALSE)
  
  # Model comparison functions
  get_model_metrics <- function(model_name, data, threshold = 0.5) {
    if (is.null(data)) return(NULL)
    
    direction <- model_score_directions[[model_name]]
    if (is.null(direction)) direction <- "gte"
    
    metrics <- calculate_metrics(data$true_label, data$predicted_score, threshold, direction)
    
    return(data.frame(
      Model = model_name,
      Threshold = threshold,
      Precision = metrics$precision,
      Recall = metrics$recall,
      F1_Score = metrics$f1_score,
      MCC = metrics$mcc,
      FPR = metrics$fpr_at_threshold,
      TPR = metrics$tpr_at_threshold,
      Specificity = metrics$specificity
    ))
  }
  
  output$model_comparison_plot <- renderPlot({
    # Get all available models with proper mapping
    models_list <- list()
    model_names <- c()
    
    if (!is.null(xgb_preds)) {
      models_list[["XGBoost"]] <- xgb_preds
      model_names <- c(model_names, "XGBoost")
    }
    if (!is.null(rf_preds)) {
      models_list[["Random Forest"]] <- rf_preds
      model_names <- c(model_names, "Random Forest")
    }
    if (!is.null(lr_preds)) {
      models_list[["Logistic Regression"]] <- lr_preds
      model_names <- c(model_names, "Logistic Regression")
    }
    
    if (length(models_list) == 0) {
      return(ggplot() + annotate("text", x=0.5, y=0.5, label="No supervised models available for comparison.") + theme_void())
    }
    
    # Calculate metrics for each model at default threshold
    comparison_data <- data.frame()
    
    for (model_name in names(models_list)) {
      # Map display names to internal keys
      model_key <- switch(model_name,
        "XGBoost" = "xgb",
        "Random Forest" = "rf", 
        "Logistic Regression" = "lr",
        "demo"  # fallback
      )
      
      threshold <- optimal_f1_thresholds[[model_key]]
      if (is.null(threshold)) threshold <- 0.5
      
      metrics <- get_model_metrics(model_key, models_list[[model_name]], threshold)
      if (!is.null(metrics)) {
        comparison_data <- rbind(comparison_data, metrics)
      }
    }
    
    if (nrow(comparison_data) == 0) {
      return(ggplot() + annotate("text", x=0.5, y=0.5, label="No data available for comparison.") + theme_void())
    }
    
    # Create comparison plot
    p <- ggplot(comparison_data, aes(x = Model, y = F1_Score, fill = Model)) +
      geom_bar(stat = "identity", alpha = 0.8) +
      geom_text(aes(label = round(F1_Score, 3)), vjust = -0.5, size = 4) +
      labs(title = "Model Performance Comparison (F1-Score)", 
           subtitle = "Using optimal thresholds",
           y = "F1-Score", x = "Model") +
      scale_fill_brewer(palette = "Set1") +
      theme_minimal() +
      theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))
    
    return(p)
  })
  
  output$model_comparison_table <- renderTable({
    # Get all available models with proper mapping
    models_list <- list()
    
    if (!is.null(xgb_preds)) {
      models_list[["XGBoost"]] <- xgb_preds
    }
    if (!is.null(rf_preds)) {
      models_list[["Random Forest"]] <- rf_preds
    }
    if (!is.null(lr_preds)) {
      models_list[["Logistic Regression"]] <- lr_preds
    }
    
    if (length(models_list) == 0) {
      return(data.frame(Message = "No supervised models available for comparison."))
    }
    
    # Calculate metrics for each model at default threshold
    comparison_data <- data.frame()
    
    for (model_name in names(models_list)) {
      # Map display names to internal keys
      model_key <- switch(model_name,
        "XGBoost" = "xgb",
        "Random Forest" = "rf", 
        "Logistic Regression" = "lr",
        "demo"  # fallback
      )
      
      threshold <- optimal_f1_thresholds[[model_key]]
      if (is.null(threshold)) threshold <- 0.5
      
      metrics <- get_model_metrics(model_key, models_list[[model_name]], threshold)
      if (!is.null(metrics)) {
        comparison_data <- rbind(comparison_data, metrics)
      }
    }
    
    if (nrow(comparison_data) == 0) {
      return(data.frame(Message = "No data available for comparison."))
    }
    
    # Format for display
    display_data <- comparison_data[, c("Model", "Threshold", "Precision", "Recall", "F1_Score", "MCC", "Specificity")]
    display_data$Precision <- round(display_data$Precision, 4)
    display_data$Recall <- round(display_data$Recall, 4)
    display_data$F1_Score <- round(display_data$F1_Score, 4)
    display_data$MCC <- round(display_data$MCC, 4)
    display_data$Threshold <- round(display_data$Threshold, 4)
    display_data$Specificity <- round(display_data$Specificity, 4)
    
    return(display_data)
  }, rownames = FALSE)
  
  # Advanced metrics plots for imbalanced classification
  output$threshold_analysis_plot <- renderPlot({
    req(input$model_select, input$threshold)
    data <- selected_data()
    
    if (is.null(data) || nrow(data) == 0) {
      return(ggplot() + annotate("text", x=0.5, y=0.5, label="No data available for threshold analysis.") + theme_void())
    }
    
    direction <- model_score_directions[[input$model_select]]
    
    # Generate thresholds (simplified for performance)
    scores <- data$predicted_score
    thresholds <- seq(min(scores, na.rm = TRUE), max(scores, na.rm = TRUE), length.out = 50)
    
    # Calculate metrics at each threshold
    metrics_df <- data.frame()
    for (thresh in thresholds) {
      metrics <- calculate_metrics(data$true_label, scores, thresh, direction)
      metrics_df <- rbind(metrics_df, data.frame(
        threshold = thresh,
        precision = metrics$precision,
        recall = metrics$recall,
        f1_score = metrics$f1_score
      ))
    }
    
    if (nrow(metrics_df) == 0) {
      return(ggplot() + annotate("text", x=0.5, y=0.5, label="No data available for threshold analysis.") + theme_void())
    }
    
    # Create threshold analysis plot
    p <- ggplot(metrics_df, aes(x = threshold)) +
      geom_line(aes(y = precision, color = "Precision"), linewidth = 1) +
      geom_line(aes(y = recall, color = "Recall"), linewidth = 1) +
      geom_line(aes(y = f1_score, color = "F1-Score"), linewidth = 1) +
      geom_vline(xintercept = input$threshold, linetype = "dashed", color = "red", linewidth = 1) +
      annotate("text", x = input$threshold, y = 0.5,
               label = paste0("Current: ", round(input$threshold, 3)), 
               vjust = -0.5, color = "red") +
      labs(title = "Metrics vs Threshold", 
           subtitle = paste("Data points:", nrow(metrics_df), "| Score range:", round(min(scores), 3), "to", round(max(scores), 3)),
           x = "Threshold", y = "Metric Value", color = "Metric") +
      scale_color_manual(values = c("Precision" = "#E41A1C", "Recall" = "#377EB8", "F1-Score" = "#4DAF4A")) +
      theme_minimal() +
      theme(legend.position = "top")
    
    return(p)
  })
  
  output$cumulative_gain_plot <- renderPlot({
    req(input$model_select, input$threshold)
    data <- selected_data()
    
    if (is.null(data) || nrow(data) == 0) {
      return(ggplot() + annotate("text", x=0.5, y=0.5, label="No data available for cumulative gain plot.") + theme_void())
    }
    
    # Sort by predicted score (descending for fraud detection)
    sorted_data <- data[order(-data$predicted_score), ]
    n_total <- nrow(sorted_data)
    n_fraud <- sum(sorted_data$true_label == 1)
    
    if (n_fraud == 0) {
      return(ggplot() + annotate("text", x=0.5, y=0.5, label="No fraud cases found in data.") + theme_void())
    }
    
    # Calculate cumulative metrics
    cumulative_fraud <- cumsum(sorted_data$true_label == 1)
    cumulative_percent <- (1:n_total) / n_total * 100
    gain_percent <- cumulative_fraud / n_fraud * 100
    
    # Create gain chart data
    gain_data <- data.frame(
      Percentile = cumulative_percent,
      Gain = gain_percent,
      Baseline = cumulative_percent
    )
    
    p <- ggplot(gain_data, aes(x = Percentile)) +
      geom_line(aes(y = Gain, color = "Model"), linewidth = 1.5) +
      geom_line(aes(y = Baseline, color = "Random"), linewidth = 1, linetype = "dashed") +
      geom_abline(intercept = 0, slope = 1, linetype = "dotted", color = "grey") +
      labs(title = "Cumulative Gain Chart", 
           subtitle = "How well the model concentrates fraud predictions",
           x = "Percentile of Transactions (%)", 
           y = "Cumulative % of Frauds Found (%)",
           color = "Strategy") +
      scale_color_manual(values = c("Model" = "#E41A1C", "Random" = "#377EB8")) +
      theme_minimal() +
      theme(legend.position = "top")
    
    return(p)
  })
  
  output$class_wise_metrics_plot <- renderPlot({
    req(input$threshold, input$model_select)
    data <- selected_data()
    
    if (is.null(data) || nrow(data) == 0) {
      return(ggplot() + annotate("text", x=0.5, y=0.5, label="No data available for class-wise metrics.") + theme_void())
    }
    
    direction <- model_score_directions[[input$model_select]]
    metrics <- calculate_metrics(data$true_label, data$predicted_score, input$threshold, direction)
    
    # Calculate class-wise metrics
    cm <- metrics$confusion_matrix
    tp <- cm[2, 2]
    tn <- cm[1, 1]
    fp <- cm[1, 2]
    fn <- cm[2, 1]
    
    # Calculate additional metrics
    specificity <- ifelse(tn + fp == 0, 0, tn / (tn + fp))  # True Negative Rate (TN / (TN + FP))
    sensitivity <- ifelse(tp + fn == 0, 0, tp / (tp + fn))  # Same as recall/TPR
    precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    recall <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    f1_score <- ifelse(precision + recall == 0, 0, 2 * (precision * recall) / (precision + recall))
    
    # Create metrics data frame
    metrics_data <- data.frame(
      Metric = c("Precision", "Recall (Sensitivity)", "Specificity", "F1-Score"),
      Value = c(precision, recall, specificity, f1_score),
      Class = c("Fraud", "Fraud", "Normal", "Overall")
    )
    
    p <- ggplot(metrics_data, aes(x = Metric, y = Value, fill = Class)) +
      geom_bar(stat = "identity", alpha = 0.8) +
      geom_text(aes(label = round(Value, 3)), vjust = -0.5, size = 4) +
      labs(title = "Class-wise Performance Metrics", 
           subtitle = paste("At threshold:", round(input$threshold, 3)),
           x = "Metric", y = "Value", fill = "Class") +
      scale_fill_manual(values = c("Fraud" = "#E41A1C", "Normal" = "#377EB8", "Overall" = "#4DAF4A"), 
                       limits = c("Fraud", "Normal", "Overall")) +
      theme_minimal() +
      theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1)) +
      ylim(0, 1)
    
    return(p)
  })
  
  output$calibration_plot <- renderPlot({
    req(input$model_select, input$threshold)
    data <- selected_data()
    
    if (is.null(data) || nrow(data) == 0) {
      return(ggplot() + annotate("text", x=0.5, y=0.5, label="No data available for calibration plot.") + theme_void())
    }
    
    # Create calibration bins
    n_bins <- 10
    data$bin <- cut(data$predicted_score, breaks = n_bins, labels = FALSE)
    
    # Calculate mean predicted probability and actual fraction for each bin
    calibration_data <- data %>%
      group_by(bin) %>%
      summarise(
        mean_pred = mean(predicted_score, na.rm = TRUE),
        actual_fraction = mean(true_label, na.rm = TRUE),
        count = n()
      ) %>%
      filter(!is.na(bin))
    
    if (nrow(calibration_data) == 0) {
      return(ggplot() + annotate("text", x=0.5, y=0.5, label="Not enough data for calibration plot.") + theme_void())
    }
    
    p <- ggplot(calibration_data, aes(x = mean_pred, y = actual_fraction)) +
      geom_point(aes(size = count), alpha = 0.7, color = "#E41A1C") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey") +
      geom_smooth(method = "loess", se = TRUE, color = "#377EB8", alpha = 0.3, formula = y ~ x) +
      labs(title = "Calibration Plot", 
           subtitle = "How well predicted probabilities match actual fraud rates",
           x = "Mean Predicted Probability", 
           y = "Actual Fraction of Frauds",
           size = "Sample Count") +
      theme_minimal() +
      theme(legend.position = "top")
    
    return(p)
  })
  
  output$ks_plot <- renderPlot({
    req(input$model_select, input$threshold)
    data <- selected_data()
    
    if (is.null(data) || nrow(data) == 0) {
      return(ggplot() + annotate("text", x=0.5, y=0.5, label="No data available for KS plot.") + theme_void())
    }
    
    # Calculate cumulative distributions
    fraud_scores <- data$predicted_score[data$true_label == 1]
    normal_scores <- data$predicted_score[data$true_label == 0]
    
    if (length(fraud_scores) == 0 || length(normal_scores) == 0) {
      return(ggplot() + annotate("text", x=0.5, y=0.5, label="Not enough data for KS plot.") + theme_void())
    }
    
    # Create cumulative distribution data
    score_range <- seq(min(data$predicted_score, na.rm = TRUE), max(data$predicted_score, na.rm = TRUE), length.out = 100)
    
    cdf_fraud <- sapply(score_range, function(x) mean(fraud_scores <= x, na.rm = TRUE))
    cdf_normal <- sapply(score_range, function(x) mean(normal_scores <= x, na.rm = TRUE))
    
    ks_data <- data.frame(
      score = score_range,
      fraud_cdf = cdf_fraud,
      normal_cdf = cdf_normal,
      difference = cdf_fraud - cdf_normal
    )
    
    # Find KS statistic
    ks_stat <- max(abs(ks_data$difference), na.rm = TRUE)
    ks_threshold <- ks_data$score[which.max(abs(ks_data$difference))]
    
    p <- ggplot(ks_data, aes(x = score)) +
      geom_line(aes(y = fraud_cdf, color = "Fraud"), linewidth = 1) +
      geom_line(aes(y = normal_cdf, color = "Normal"), linewidth = 1) +
      geom_line(aes(y = difference, color = "Difference"), linewidth = 1, linetype = "dashed") +
      geom_vline(xintercept = ks_threshold, linetype = "dotted", color = "red") +
      annotate("text", x = ks_threshold, y = 0.5,
               label = paste0("KS = ", round(ks_stat, 3)), 
               vjust = -0.5, color = "red") +
      labs(title = "Kolmogorov-Smirnov Statistic Plot", 
           subtitle = paste("KS Statistic =", round(ks_stat, 3)),
           x = "Predicted Score", 
           y = "Cumulative Probability",
           color = "Distribution") +
      scale_color_manual(values = c("Fraud" = "#E41A1C", "Normal" = "#377EB8", "Difference" = "#4DAF4A")) +
      theme_minimal() +
      theme(legend.position = "top")
    
    return(p)
  })
  
  output$score_distribution_by_class_plot <- renderPlot({
    req(input$model_select, input$threshold)
    data <- selected_data()
    
    if (is.null(data) || nrow(data) == 0) {
      return(ggplot() + annotate("text", x=0.5, y=0.5, label="No data available for score distribution plot.") + theme_void())
    }
    
    # Create violin plot with box plot overlay
    p <- ggplot(data, aes(x = factor(true_label, labels = c("Normal", "Fraud")), y = predicted_score, fill = factor(true_label))) +
      geom_violin(alpha = 0.7) +
      geom_boxplot(width = 0.2, alpha = 0.8, fill = "white") +
      geom_hline(yintercept = input$threshold, linetype = "dashed", color = "red", linewidth = 1) +
      annotate("text", x = 1.5, y = input$threshold,
               label = paste0("Threshold: ", round(input$threshold, 3)), 
               vjust = -0.5, color = "red") +
      labs(title = "Score Distribution by Class", 
           subtitle = "Violin plot with box plot overlay",
           x = "True Class", 
           y = "Predicted Score",
           fill = "Class") +
      scale_fill_manual(values = c("Normal" = "#377EB8", "Fraud" = "#E41A1C"), 
                       limits = c("Normal", "Fraud")) +
      theme_minimal() +
      theme(legend.position = "none")
    
    return(p)
  })
  
  output$metrics_warning <- renderUI({
    req(input$model_select)
    if (input$model_select %in% c("if", "lof", "ae")) {
      tags$div(
        class = "alert alert-warning",
        role = "alert",
        "⚠️ Warning: Unsupervised Model",
        p("These metrics assume anomaly scores correspond to fraud labels."),
        p("This may not be accurate for unsupervised models.")
      )
    }
  })
  

  
  output$readme_content <- renderUI({
    if (file.exists("README.md")) {
      includeMarkdown("README.md")
    } else {
      h3("README.md not found.")
    }
  })
}

# --- Run the App ---
shinyApp(ui, server)