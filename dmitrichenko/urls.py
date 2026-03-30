from django.urls import path
from dmitrichenko import views

urlpatterns = [
    path('', views.index, name='home'),         # Главная: http://127.0.0.1:8000
    path('about/', views.about, name='about_page'),   # О студенте: http://127.0.0.1:8000/about/
    path('data/', views.data_view, name='data_page'),     # Данные: http://127.0.0.1:8000/data/
    path('graph/', views.graph_view, name='graph_page'),
    path('userform/', views.user_form_view, name='userform'),
]