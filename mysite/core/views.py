from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from django.urls import reverse_lazy

from .forms import BookForm
from .models import Book
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import os
import pydotplus  

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 

class Home(TemplateView):
    template_name = 'home.html'


def upload(request):
    context = {}
    upload_file_name = ""
    image_url = ""
  
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        image_url = "Tree_of_"+str(os.path.splitext(uploaded_file.name)[0]) + ".png" 
        dataset_cols_name = []
        pima = pd.read_csv(uploaded_file , header=0)
        dataset_cols_name = pima.columns.values.tolist()
        transet_cols_name  = dataset_cols_name[:len(dataset_cols_name)-1]
        transet_cols_name.append("decisionCol")
        pima.columns = transet_cols_name
        
        #split dataset in features and target variable
        feature_cols = transet_cols_name[:len(transet_cols_name)-1]
        X = pima[feature_cols] # Features
        y = pima.decisionCol # Target variable

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier()

        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        #Predict the response for test dataset
        y_pred = clf.predict(X_test)


        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data,  
                        filled=True, rounded=True,
                        special_characters=True,feature_names = feature_cols,class_names=['0','1'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
        graph.write_png(image_url)
        imge = Image(graph.create_png())
        
        fs = FileSystemStorage()
        upload_file_name = uploaded_file
        name = fs.save(uploaded_file.name, uploaded_file)
        path_to_generated_image = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+"\\django-upload-example\\"+ image_url
        print(path_to_generated_image)
        context['url'] = path_to_generated_image
        

    #fileUrl = os.path.join(BASE_DIR , '\\media\\' + upload_file_name)
    #
    #print(type(image))
    return render(request, 'upload.html', context)


def book_list(request):
    books = Book.objects.all()
    return render(request, 'book_list.html', {
        'books': books
    })



def upload_book(request):
    if request.method == 'POST':
        form = BookForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('book_list')
    else:
        form = BookForm()
    return render(request, 'upload_book.html', {
        'form': form
    })


def delete_book(request, pk):
    if request.method == 'POST':
        book = Book.objects.get(pk=pk)
        book.delete()
    return redirect('book_list')


class BookListView(ListView):
    model = Book
    template_name = 'class_book_list.html'
    context_object_name = 'books'


class UploadBookView(CreateView):
    model = Book
    form_class = BookForm
    success_url = reverse_lazy('class_book_list')
    template_name = 'upload_book.html'
