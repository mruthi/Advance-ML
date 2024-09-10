from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, redirect
from django.contrib import messages
from django.urls import reverse_lazy
from django.views.generic import View
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Assuming ckdForm is a valid form you have defined elsewhere in your code.
from .forms import AdForm

np.random.seed(123)  # Ensure reproducibility

class dataUploadView(View):
    form_class = AdForm
    success_url = reverse_lazy('success')
    template_name = 'create.html'
    failure_url = reverse_lazy('fail')
    filenot_url = reverse_lazy('filenot')

    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            data_age = request.POST.get('Age')
            data_es = request.POST.get('EstimatedSalary')
            data_gm = request.POST.get('Gender_Male')
            import pandas as pd
            dataset=pd.read_csv("Social_Network_Ads.csv")
            dataset=pd.get_dummies(dataset,drop_first=True)
            dataset.drop("User ID",axis=1)
            independent=dataset[[ 'Age', 'EstimatedSalary','Gender_Male']]
            dependent=dataset[['Purchased']]
            from sklearn.model_selection import train_test_split
            x_train,x_test,y_train,y_test=train_test_split(independent,dependent,test_size=1/3,random_state=0)
            from sklearn.naive_bayes import CategoricalNB
            classifier=CategoricalNB()
            classifier.fit(x_train,y_train)
            y_pred=classifier.predict(x_test)
            from sklearn.metrics import confusion_matrix
            cm=confusion_matrix(y_pred,y_test)
            from sklearn.metrics import classification_report
            clf=classification_report(y_pred,y_test)

            # Convert data to numeric and reshape for prediction
            data = np.array([float(data_age), float(data_es), float(data_gm)])
            if data.dtype.kind in 'UO':
                data = data.astype(float)
            # Predict the outcome
            out = classifier.predict(data.reshape(1, -1))

            # Render the result in a template
            return render(request, "succ_msg.html", {
                'data_age': data_age,
                'data_es': data_es,
                'data_gm': data_gm,
                'out': out
            })
        else:
            return redirect(self.failure_url)
