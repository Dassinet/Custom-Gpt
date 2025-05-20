import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { axiosInstance, setAccessToken, getAccessToken, removeAccessToken } from '../api/axiosInstance';
import axios from 'axios';


const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [accessToken, _setAccessToken] = useState(() => getAccessToken());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();


  const updateAccessToken = useCallback((token) => {
    if (token) {
      setAccessToken(token);
      _setAccessToken(token);
    } else {
      removeAccessToken();
      _setAccessToken(null);
    }
  }, []);


  const fetchUser = useCallback(async () => {
    const currentToken = getAccessToken();
    if (!currentToken) {
      const savedUser = localStorage.getItem('user');
      if (savedUser) {
        try {
          setUser(JSON.parse(savedUser));
          const response = await axiosInstance.post('/api/auth/refresh');
          if (response.data && response.data.accessToken) {
            updateAccessToken(response.data.accessToken);
          }
        } catch (e) {
          console.error("Failed to use saved user data:", e);
          setUser(null);
          localStorage.removeItem('user');
        }
      } else {
        setUser(null);
      }
      setLoading(false);
      return;
    }

    setLoading(true);
    try {
      const response = await axiosInstance.get('/api/auth/me');
      if (response.data) {
        setUser(response.data);
        localStorage.setItem('user', JSON.stringify(response.data));
      } else {
        updateAccessToken(null);
        setUser(null);
        localStorage.removeItem('user');
      }
    } catch (err) {
      console.error("Error fetching user:", err);
      updateAccessToken(null);
      setUser(null);
      localStorage.removeItem('user');
    } finally {
      setLoading(false);
    }
  }, [updateAccessToken]);


  useEffect(() => {
    const checkAuth = async () => {
      const token = getAccessToken();
      if (token) {
        try {
          const response = await axiosInstance.get('/api/auth/me');
          setUser(response.data);
        } catch (err) {
          removeAccessToken();
          setUser(null);
        }
      }
      setLoading(false);
    };
    
    checkAuth();
  }, []);

  const handleAuthCallback = useCallback((token, userData) => {
    updateAccessToken(token);
    setUser(userData);
    setLoading(false);


    localStorage.setItem('user', JSON.stringify(userData));

    if (userData?.role === 'admin') {
      navigate('/admin', { replace: true });
    } else {
      navigate('/employee', { replace: true });
    }
  }, [navigate, updateAccessToken]);


  const login = async (email, password) => {
    if (!email || !password) {
      setError('Email and password are required.');
      return;
    }
    try {
      setLoading(true);
      setError(null);
      const response = await axiosInstance.post('/api/auth/login', { email, password });
      setUser(response.data.user);
      updateAccessToken(response.data.accessToken);
      return response.data;
    } catch (err) {
      const errorMessage = err.response?.data?.message || 'Login failed. Please try again.';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const signup = async (name, email, password) => {
    try {
      setLoading(true);
      setError(null);
      const response = await axiosInstance.post('/api/auth/register', { name, email, password });
      setUser(response.data.user);
      return response.data;
    } catch (err) {
      setError(err.response?.data?.message || 'Signup failed');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    try {
      await axiosInstance.post('/api/auth/logout');
    } catch (err) {
      console.error('Logout failed:', err);
    } finally {
      // Clear all auth-related storage
      removeAccessToken(); // This will clear localStorage token and axios headers
      localStorage.removeItem('user');
      sessionStorage.clear(); // Clear all session storage
      document.cookie.split(";").forEach((c) => {
        document.cookie = c
          .replace(/^ +/, "")
          .replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/");
      }); // Clear all cookies in the browser
      setUser(null);
      _setAccessToken(null);
      navigate('/login', { replace: true }); // Use replace to prevent going back
    }
  };

  const googleAuthInitiate = () => {
    window.location.href = `${axiosInstance.defaults.baseURL}/api/auth/google`;
  };

  const forgotPassword = async (email) => {
    try {
      setLoading(true);
      setError(null);
      const response = await axiosInstance.post('/api/auth/forgot-password', { email });
      return response.data;
    } catch (err) {
      const errorMessage = err.response?.data?.message || 'Failed to send password reset email';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const verifyResetToken = async (token) => {
    try {
      setLoading(true);
      const response = await axiosInstance.get(`/api/auth/verify-reset-token/${token}`);
      return response.data;
    } catch (err) {
      setError('Invalid or expired reset token');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const resetPassword = async (token, password) => {
    try {
      setLoading(true);
      setError(null);
      const response = await axiosInstance.post(`/api/auth/reset-password/${token}`, { password });
      return response.data;
    } catch (err) {
      const errorMessage = err.response?.data?.message || 'Failed to reset password';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthContext.Provider value={{
      user,
      accessToken,
      loading,
      error,
      login,
      signup,
      logout,
      googleAuthInitiate,
      handleAuthCallback,
      fetchUser,
      setError,
      forgotPassword,
      verifyResetToken,
      resetPassword
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  return useContext(AuthContext);
};
