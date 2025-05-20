import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FcGoogle } from 'react-icons/fc';
import { useAuth } from '../context/AuthContext';

const SignupPage = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { signup, loading, error, setError, googleAuthInitiate } = useAuth();

  useEffect(() => {
    setError(null);
  }, [setError]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    await signup(name, email, password);
  };

  return (
    <div className="flex h-screen w-full bg-white">
      {/* Left side - Image and Text */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-b from-[#02091A] to-[#031555] flex-col items-center relative overflow-hidden py-12">
        {/* Top Logo */}
        <div className="relative z-10 w-full flex justify-center mb-12 mt-8">
          <img 
            src="/Horizontal EMSA logo.png" 
            alt="EMSA" 
            className="w-4/5 max-h-36 object-contain"
          />
        </div>
        
        {/* Center Text Content */}
        <div className="relative z-10 flex flex-col items-center justify-center flex-grow px-12 text-center">
          <h1 className="text-5xl font-bold mb-8 text-white">Decision Intelligence Starts Here</h1>
          <p className="text-xl font-medium mb-6 text-white">Welcome to EMSA's Intelligent AI Dashboards</p>
          <p className="text-lg opacity-90 mb-6 text-[#A1B0C5]">
            Access GPT-powered assistants designed to analyse data, generate insights, and support your operational decisions in real time.
          </p>
          <p className="text-lg italic mb-8 text-[#FBFCFD]">
            From complexity to clarity—in just a few clicks.
          </p>
          <div className="h-1 w-32 bg-[#055FF7] mt-4"></div>
        </div>
        
        {/* Background gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-br from-[#031555] to-[#083EA9] opacity-20"></div>
      </div>

      {/* Right side - Signup Form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center px-6 md:px-16 py-12">
        <div className="w-full max-w-md">
          <div className="text-center mb-10">
            <h2 className="text-3xl font-bold text-gray-900 mb-2">Create Account</h2>
            <p className="text-gray-600">Please enter your details</p>
          </div>

          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4" role="alert">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
                Full Name
              </label>
              <input
                id="name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent transition-all bg-gray-50"
                placeholder="John Doe"
                required
              />
            </div>

            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent transition-all bg-gray-50"
                placeholder="john@example.com"
                required
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent transition-all bg-gray-50"
                placeholder="••••••••"
                required
              />
            </div>

            <button
              type="submit"
              className={`w-full bg-black hover:bg-gray-800 text-white py-3 rounded-lg font-medium shadow-sm transition-all duration-200 transform hover:translate-y-[-2px] ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
              disabled={loading}
            >
              {loading ? 'Signing Up...' : 'Sign Up'}
            </button>

            <div className="flex items-center my-4">
              <div className="flex-1 h-px bg-gray-300"></div>
              <p className="mx-4 text-sm text-gray-500">or</p>
              <div className="flex-1 h-px bg-gray-300"></div>
            </div>

            <div>
              <button
                type="button"
                onClick={googleAuthInitiate}
                disabled={loading}
                className="w-full flex items-center justify-center gap-3 bg-white border border-gray-300 py-3 rounded-lg font-medium text-gray-700 hover:bg-gray-50 transition-all shadow-sm disabled:opacity-50"
              >
                <FcGoogle size={20} />
                Sign up with Google
              </button>
            </div>
          </form>

          <p className="text-center mt-8 text-gray-600">
            Already have an account?{' '}
            <Link to="/login" className="text-black font-medium hover:underline">
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default SignupPage;
