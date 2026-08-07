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
        
class DoctorSelfRegisterSerializer(serializers.ModelSerializer):
    """
    Public self-registration for doctors -- distinct from
    RegisterPatientSerializer. Creates the account inactive and PENDING;
    an Admin must approve before the doctor can log in at all.
    """
    password = serializers.CharField(write_only=True, validators=[validate_password])
    license_document = serializers.ImageField(required=True)

    class Meta:
        model = User
        fields = [
            'id', 'username', 'email', 'password', 'phone_number',
            'specialty', 'license_number', 'license_document',
        ]

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password'],
            phone_number=validated_data.get('phone_number', ''),
            specialty=validated_data.get('specialty', ''),
            license_number=validated_data['license_number'],
            license_document=validated_data['license_document'],
            role=User.Role.DOCTOR,
            verification_status=User.VerificationStatus.PENDING,
            is_active=False,
        )


class DoctorApplicationSerializer(serializers.ModelSerializer):
    """For the Admin's 'pending doctor applications' review list/detail."""
    class Meta:
        model = User
        fields = [
            'id', 'username', 'email', 'phone_number', 'specialty',
            'license_number', 'license_document', 'verification_status',
            'verification_notes', 'date_joined',
        ]
        read_only_fields = fields


class ReviewDoctorApplicationSerializer(serializers.Serializer):
    """Admin's approve/reject action payload."""
    decision = serializers.ChoiceField(choices=['approve', 'reject'])
    notes = serializers.CharField(required=False, allow_blank=True)


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
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['role'] = user.role
        token['username'] = user.username
        return token

    def validate(self, attrs):
        
        
        User_ = get_user_model()
        candidate = User_.objects.filter(username=attrs.get('username')).first()
        if candidate and candidate.is_doctor() and not candidate.is_active:
            if candidate.verification_status == candidate.VerificationStatus.PENDING:
                raise serializers.ValidationError(
                    'Your doctor account is awaiting admin verification.'
                )
            if candidate.verification_status == candidate.VerificationStatus.REJECTED:
                raise serializers.ValidationError(
                    'Your doctor application was not approved. '
                    f'Reason: {candidate.verification_notes or "not specified"}.'
                )

        data = super().validate(attrs)
        data['user_id'] = self.user.id
        data['role'] = self.user.role
        data['username'] = self.user.username
        return data
