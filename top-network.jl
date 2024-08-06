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

# Seed for reproducibility
Random.seed!(1234)

# Load the cleaned dataset
data = CSV.read("./archive/final_data_top_7_genres.csv", DataFrame)

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

train_loader = Flux.Data.DataLoader((X_train', y_train'), batchsize=256, shuffle=true)
test_loader = Flux.Data.DataLoader((X_test', y_test'), batchsize=256)

println("Data loading completed")
println("Train loader size: ", length(train_loader))
println("Test loader size: ", length(test_loader))

# Model 4: Deeper Network with Dropout, BatchNorm, and ReLU Activation
model4 = Chain(
    Dense(size(X, 2), 512, relu),
    Dropout(0.5),
    BatchNorm(512),
    Dense(512, 256, relu),
    Dropout(0.5),
    BatchNorm(256),
    Dense(256, 128, relu),
    Dense(128, size(y, 2)),
    softmax
)

# Define the loss function and optimizer
loss4(x, y) = Flux.crossentropy(model4(x), y)
opt4 = Flux.Optimise.ADAM(0.02)

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
function train_model(model, loss, opt, train_loader, test_loader, epochs=50)
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

# Train the model 4 and store metrics
println("Training Model 4")
train_accuracies4, val_accuracies4, train_losses4, val_losses4, best_val_accuracy4, best_epoch4 = train_model(model4, loss4, opt4, train_loader, test_loader)

# Print the best validation accuracy and epoch for model 4
println("Best validation accuracy for Model 4: $best_val_accuracy4 at epoch $best_epoch4")

# Plotting function
function plot_metrics(epochs, train_metric, val_metric, metric_name, model_name)
    plot(epochs, train_metric, label="Train $metric_name", xlabel="Epoch", ylabel=metric_name, title="$model_name: $metric_name Over Epochs")
    plot!(epochs, val_metric, label="Val $metric_name")
    savefig("archive/results/$model_name$metric_name.png")
end

epochs = 1:50

# Plot metrics for Model 4
plot_metrics(epochs, train_accuracies4, val_accuracies4, "Accuracy", "Model4")
plot_metrics(epochs, train_losses4, val_losses4, "Loss", "Model4")
