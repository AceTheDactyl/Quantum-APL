import { Tabs } from 'expo-router';
import { Eye, Lock, Wrench } from 'lucide-react-native';
import React from 'react';

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: '#1a0b2e',
          borderTopColor: 'rgba(167, 139, 250, 0.2)',
          borderTopWidth: 1,
        },
        tabBarActiveTintColor: '#a78bfa',
        tabBarInactiveTintColor: '#6b7280',
      }}
    >
      <Tabs.Screen
        name="decode"
        options={{
          title: 'Decode',
          tabBarIcon: ({ color, size }) => <Eye size={size} color={color} />,
        }}
      />
      <Tabs.Screen
        name="encode"
        options={{
          title: 'Encode',
          tabBarIcon: ({ color, size }) => <Lock size={size} color={color} />,
        }}
      />
      <Tabs.Screen
        name="tools"
        options={{
          title: 'Tools',
          tabBarIcon: ({ color, size }) => <Wrench size={size} color={color} />,
        }}
      />
    </Tabs>
  );
}
