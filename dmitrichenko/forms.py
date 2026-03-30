from django import forms

class UserForm(forms.Form):
    name = forms.CharField(label="Имя", widget=forms.TextInput(attrs={'class': 'form-control'}))
    last_name = forms.CharField(label="Фамилия", widget=forms.TextInput(attrs={'class': 'form-control'}))
    age = forms.IntegerField(label="Возраст", min_value=1, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(label="Электронная почта", widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'example@mail.com'}))
    date = forms.DateField(label="Дата", widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}))
    recommendations = forms.CharField(
        label="Рекомендации", 
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        required=False
    )