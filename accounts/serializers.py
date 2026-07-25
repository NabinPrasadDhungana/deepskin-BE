from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

User = get_user_model()


class RegisterPatientSerializer(serializers.ModelSerializer):
    """
    Public self-registration -- ALWAYS creates a patient account.
    Doctor and admin accounts are never created through this endpoint;
    see accounts/views.py for why (only Admins may create Doctor accounts).
    """
    password = serializers.CharField(write_only=True, validators=[validate_password])

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'password', 'phone_number']

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password'],
            phone_number=validated_data.get('phone_number', ''),
            role=User.Role.PATIENT,
        )


class CreateDoctorSerializer(serializers.ModelSerializer):
    """Admin-only: create a doctor account. See accounts/views.py permissions."""
    password = serializers.CharField(write_only=True, validators=[validate_password])

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'password', 'specialty', 'phone_number']

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password'],
            specialty=validated_data.get('specialty', ''),
            phone_number=validated_data.get('phone_number', ''),
            role=User.Role.DOCTOR,
        )


class UserSerializer(serializers.ModelSerializer):
    """Read-only representation of a user -- used when nesting user info
    inside case/message responses so the frontend doesn't need extra calls."""

    class Meta:
        model = User
        fields = ['id', 'username', 'role', 'specialty']
        read_only_fields = fields


class DeepSkinTokenObtainPairSerializer(TokenObtainPairSerializer):
    """
    Adds `role` and `user_id` to the JWT payload so the frontend can route
    the user to the right dashboard immediately after login without an
    extra "who am I" API call.
    """

    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['role'] = user.role
        token['username'] = user.username
        return token

    def validate(self, attrs):
        data = super().validate(attrs)
        data['user_id'] = self.user.id
        data['role'] = self.user.role
        data['username'] = self.user.username
        return data
