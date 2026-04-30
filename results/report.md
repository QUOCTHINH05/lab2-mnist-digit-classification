# Comparison
- The Neural Network achieved higher accuracy (97.66% compare to 89.70%), higher than logistic by 7.96%
- The Neural Network has much more parameters than the Logistic model, 109386 compare to 7850.
- The more complexity a model is, the higher accuracy the model achieves.

# Struggle
Base on the confusion matrix, we can figure out that:
- With Neural Network, the model struggles the most with recognizing 4 from 9 with 22 mistakes base on the value on row 4th column 9th.
- With Logistic, the model also struggles the most with recognizing 4 from 9 and 9 from 4, with 50 and 51 mistakes repectively.
- Overall, the Neural Network is better because it makes fewer mistake predictions. 
- This shows that the extra hidden layers and non-linear activations help the model distinguish subtle differences that a simpler model misses.

# Improvement
Adding Dropout Layers between Dense layers. 

Explain: Dropout randomly turns off a percentage of neurons during each training step. This prevents the model from relying too heavily on specific pixels (which reduces overfitting) and forces the network to learn more robust, general features of the digits.