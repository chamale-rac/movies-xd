# Import necessary packages
using Pkg
Pkg.add(["Flux", "CSV", "DataFrames", "Statistics", "MLUtils", "Random", "Plots"])

using Flux
using CSV
using DataFrames
using Statistics
using MLUtils
using Random
using LinearAlgebra
using Plots

Random.seed!(1234)

# Load the cleaned dataset
data = CSV.read("./archive/final_data.csv", DataFrame)

# Split the data into features and target
X = select(data, Not(r"adventure|drama|family|action|comedy|fantasy|animation|sci-fi|thriller|war|crime|history|biography|sport|romance|horror|musical|mystery|western|music|documentary|film-noir|news")) |> Matrix
y = select(data, r"adventure|drama|family|action|comedy|fantasy|animation|sci-fi|thriller|war|crime|history|biography|sport|romance|horror|musical|mystery|western|music|documentary|film-noir|news") |> Matrix

# Function to partition the data
function partitionTrainTest(data, at = 0.7)
    n = nrow(data)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx, :], data[test_idx, :]
end
# Split Reference: https://discourse.julialang.org/t/simple-tool-for-train-test-split/473/2

# Apply partitioning function
train_data, test_data = partitionTrainTest(data, 0.8)

# Split train and test data into features and targets
X_train = select(train_data, Not(r"adventure|drama|family|action|comedy|fantasy|animation|sci-fi|thriller|war|crime|history|biography|sport|romance|horror|musical|mystery|western|music|documentary|film-noir|news")) |> Matrix
y_train = select(train_data, r"adventure|drama|family|action|comedy|fantasy|animation|sci-fi|thriller|war|crime|history|biography|sport|romance|horror|musical|mystery|western|music|documentary|film-noir|news") |> Matrix
X_test = select(test_data, Not(r"adventure|drama|family|action|comedy|fantasy|animation|sci-fi|thriller|war|crime|history|biography|sport|romance|horror|musical|mystery|western|music|documentary|film-noir|news")) |> Matrix
y_test = select(test_data, r"adventure|drama|family|action|comedy|fantasy|animation|sci-fi|thriller|war|crime|history|biography|sport|romance|horror|musical|mystery|western|music|documentary|film-noir|news") |> Matrix

# Ensure dimensions match
println("X_train size: ", size(X_train))
println("y_train size: ", size(y_train))
println("X_test size: ", size(X_test))
println("y_test size: ", size(y_test))

train_loader = Flux.Data.DataLoader((X_train', y_train'), batchsize=32, shuffle=true)
test_loader = Flux.Data.DataLoader((X_test', y_test'), batchsize=32)

println("Data loading completed")
println("Train loader size: ", length(train_loader))
println("Test loader size: ", length(test_loader))

# Model 1: Simple Network with ReLU Activation and Dropout Regularization
model1 = Chain(
    Dense(size(X, 2), 128, relu),
    Dropout(0.3),
    Dense(128, 64, relu),
    Dropout(0.3),
    Dense(64, size(y, 2)),
    softmax
)

# Define the loss function and optimizer
loss1(x, y) = Flux.crossentropy(model1(x), y)
opt1 = ADAM(0.001)  # Adjusted learning rate

# Model 2: Deeper Network with ReLU Activation and L2 Regularization
model2 = Chain(
    Dense(size(X, 2), 256, relu),
    Dense(256, 128, relu),
    Dense(128, 64, relu),
    Dense(64, size(y, 2)),
    softmax
)

# Define the L2 regularization term
function l2_penalty(model)
    lambda = 0.0001  # Adjusted lambda
    return lambda * sum(norm, Flux.params(model))
end
# L2 Reference https://fluxml.ai/Flux.jl/previews/PR1472/models/regularisation/

# Define the loss function with L2 regularization
loss2(x, y) = Flux.logitcrossentropy(model2(x), y) + l2_penalty(model2)
opt2 = ADAM(0.001)  # Using ADAM optimizer with adjusted learning rate

# Model 3: Wider Network with Tanh Activation and Batch Normalization
model3 = Chain(
    Dense(size(X, 2), 512, tanh),
    BatchNorm(512, relu),
    Dense(512, 256, tanh),
    BatchNorm(256, relu),
    Dense(256, size(y, 2)),
    softmax
)

# Define the loss function and optimizer
loss3(x, y) = Flux.crossentropy(model3(x), y)
opt3 = ADAM(0.001)  # Adjusted learning rate

# Function to evaluate the models
function evaluate_model(model, test_loader)
    y_preds = []
    y_tests = []
    total_loss = 0.0
    for (x_batch, y_batch) in test_loader
        y_pred = model(x_batch)
        loss = Flux.crossentropy(y_pred, y_batch)
        total_loss += loss
        push!(y_preds, y_pred)
        push!(y_tests, y_batch)
    end
    y_pred = reduce(hcat, y_preds)
    y_test = reduce(hcat, y_tests)
    accuracy = sum(argmax(y_pred, dims=1) .== argmax(y_test, dims=1)) / size(y_test, 2)
    avg_loss = total_loss / length(test_loader)
    return accuracy, avg_loss
end

# Function to train the models and store metrics
function train_model(model, loss, opt, train_loader, test_loader, epochs=20)
    train_accuracies = Float64[]
    val_accuracies = Float64[]
    train_losses = Float64[]
    val_losses = Float64[]

    best_val_accuracy = 0.0
    best_epoch = 0

    for epoch in 1:epochs
        epoch_loss = 0.0
        for (x_batch, y_batch) in train_loader
            Flux.train!(loss, Flux.params(model), [(x_batch, y_batch)], opt)
            epoch_loss += loss(x_batch, y_batch)
        end
        epoch_loss /= length(train_loader)
        train_accuracy, train_loss = evaluate_model(model, train_loader)
        val_accuracy, val_loss = evaluate_model(model, test_loader)
        push!(train_accuracies, train_accuracy)
        push!(val_accuracies, val_accuracy)
        push!(train_losses, train_loss)
        push!(val_losses, val_loss)
        println("Epoch $epoch completed - Train Accuracy: $train_accuracy, Val Accuracy: $val_accuracy, Train Loss: $train_loss, Val Loss: $val_loss")

        if val_accuracy > best_val_accuracy
            best_val_accuracy = val_accuracy
            best_epoch = epoch
        end
    end

    return train_accuracies, val_accuracies, train_losses, val_losses, best_val_accuracy, best_epoch
end

# Train the models and store metrics
println("Training Model 1")
train_accuracies1, val_accuracies1, train_losses1, val_losses1, best_val_accuracy1, best_epoch1 = train_model(model1, loss1, opt1, train_loader, test_loader)
println("Training Model 2")
train_accuracies2, val_accuracies2, train_losses2, val_losses2, best_val_accuracy2, best_epoch2 = train_model(model2, loss2, opt2, train_loader, test_loader)
println("Training Model 3")
train_accuracies3, val_accuracies3, train_losses3, val_losses3, best_val_accuracy3, best_epoch3 = train_model(model3, loss3, opt3, train_loader, test_loader)

# Print the best validation accuracy and epoch for each model
println("Best validation accuracy for Model 1: $best_val_accuracy1 at epoch $best_epoch1")
println("Best validation accuracy for Model 2: $best_val_accuracy2 at epoch $best_epoch2")
println("Best validation accuracy for Model 3: $best_val_accuracy3 at epoch $best_epoch3")

# Plotting function
function plot_metrics(epochs, train_metric, val_metric, metric_name, model_name)
    plot(epochs, train_metric, label="Train $metric_name", xlabel="Epoch", ylabel=metric_name, title="$model_name: $metric_name Over Epochs")
    plot!(epochs, val_metric, label="Val $metric_name")
    savefig("archive/results/$model_name$metric_name.png")
end

epochs = 1:20

# Plot metrics for Model 1
plot_metrics(epochs, train_accuracies1, val_accuracies1, "Accuracy", "Model1")
plot_metrics(epochs, train_losses1, val_losses1, "Loss", "Model1")

# Plot metrics for Model 2
plot_metrics(epochs, train_accuracies2, val_accuracies2, "Accuracy", "Model2")
plot_metrics(epochs, train_losses2, val_losses2, "Loss", "Model2")

# Plot metrics for Model 3
plot_metrics(epochs, train_accuracies3, val_accuracies3, "Accuracy", "Model3")
plot_metrics(epochs, train_losses3, val_losses3, "Loss", "Model3")

# Plot all models' accuracies in one graph (val accuracy)
plot(epochs, val_accuracies1, label="Model1", xlabel="Epoch", ylabel="Accuracy", title="Validation Accuracy Over Epochs")
plot!(epochs, val_accuracies2, label="Model2")
plot!(epochs, val_accuracies3, label="Model3")

savefig("archive/results/ValAccuracy.png")
