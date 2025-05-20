import React, { createContext, useState, useContext, useEffect } from 'react';

const ThemeContext = createContext();

export const ThemeProvider = ({ children }) => {
  // Check if user has a theme preference stored in localStorage
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const savedTheme = localStorage.getItem('theme');
    // If theme is saved in localStorage, use it, otherwise use system preference
    if (savedTheme) {
      return savedTheme === 'dark';
    }
    // Check for system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  // Function to toggle theme
  const toggleTheme = () => {
    setIsDarkMode(prevMode => !prevMode);
  };

  // Update document class and localStorage when theme changes
  useEffect(() => {
    // Update localStorage
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
    
    // Update document class for Tailwind dark mode
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  return (
    <ThemeContext.Provider value={{ isDarkMode, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

// Custom hook for using the theme context
export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}; 