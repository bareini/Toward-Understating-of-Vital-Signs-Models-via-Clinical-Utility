<img align="right" width="100" height="100" src="noun_Heart_177835.png">


# Toward Understating of Vital Signs Models via Clinical Utility

Project by Bar Eini-Porat and Rom Gutman

In acute medical settings, vital signs are gathered routinely to monitor patients’ condition.
These signals are often the first indication for various critical events. For this reason, many
works in the field of Machine Learning (ML) engage in the prediction on such physiological
signals; some have already shown promising results (Faust et al., 2018). Yet, deployment of ML models in the clinical domain requires a careful evaluation of their performance.

In our work, we aim to build on utility measures and evaluation metrics formulated for vital signs predictions. We construct comprehensive model behavior summaries to convey vital signs prediction models’ behavior with medical context. This work proposes two types of ML behviorals summaries that can be complemantry: quantitative and visual summaries.

## Quantitative Summary Measures

To generate a comprehensive behavioral report, we formulate performance measures per clinical aspects - normal range and deviation from patient trend. In most cases, each error type leads different consequences; therefore, we define measures according to the different errors as well. As a result, two pre-defined components translate to four measures in the quantitative summary report.

## Visual Summary
Inspired by (Amir and Amir, 2018), we explore the contribution of visual summaries of time series models’ to the understanding their behavior.

