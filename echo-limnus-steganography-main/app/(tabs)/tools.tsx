import { LinearGradient } from 'expo-linear-gradient';

import { Calculator, Palette, Wrench } from 'lucide-react-native';
import React, { useState } from 'react';
import {
  Alert,
  Animated,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

interface CapacityResult {
  width: number;
  height: number;
  totalPixels: number;
  totalBits: number;
  maxPayloadBytes: number;
  maxPayloadChars: number;
}

export default function ToolsScreen() {
  const insets = useSafeAreaInsets();
  const [width, setWidth] = useState<string>('');
  const [height, setHeight] = useState<string>('');
  const [capacityResult, setCapacityResult] = useState<CapacityResult | null>(null);
  const [coverPattern, setCoverPattern] = useState<'solid' | 'gradient' | 'noise'>('gradient');
  const [coverColor, setCoverColor] = useState<string>('#1a0b2e');
  const [generatedCover, setGeneratedCover] = useState<string | null>(null);
  const [fadeAnim] = useState(new Animated.Value(0));

  const calculateCapacity = () => {
    const w = parseInt(width);
    const h = parseInt(height);

    if (isNaN(w) || isNaN(h) || w <= 0 || h <= 0) {
      Alert.alert('Error', 'Please enter valid dimensions');
      return;
    }

    const totalPixels = w * h;
    const bitsPerPixel = 3;
    const totalBits = totalPixels * bitsPerPixel;
    const headerBits = 128;
    const crcBits = 32;
    const maxPayloadBits = totalBits - headerBits - crcBits;
    const maxPayloadBytes = Math.floor(maxPayloadBits / 8);
    const maxPayloadChars = maxPayloadBytes;

    setCapacityResult({
      width: w,
      height: h,
      totalPixels,
      totalBits,
      maxPayloadBytes,
      maxPayloadChars,
    });

    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 600,
      useNativeDriver: true,
    }).start();
  };

  const generateCover = async () => {
    const w = parseInt(width);
    const h = parseInt(height);

    if (isNaN(w) || isNaN(h) || w <= 0 || h <= 0) {
      Alert.alert('Error', 'Please enter valid dimensions');
      return;
    }

    if (w > 2048 || h > 2048) {
      Alert.alert('Error', 'Maximum dimensions are 2048x2048');
      return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      Alert.alert('Error', 'Failed to create canvas');
      return;
    }

    if (coverPattern === 'solid') {
      ctx.fillStyle = coverColor;
      ctx.fillRect(0, 0, w, h);
    } else if (coverPattern === 'gradient') {
      const gradient = ctx.createLinearGradient(0, 0, w, h);
      gradient.addColorStop(0, coverColor);
      gradient.addColorStop(1, adjustBrightness(coverColor, 40));
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, w, h);
    } else if (coverPattern === 'noise') {
      const imageData = ctx.createImageData(w, h);
      const data = imageData.data;
      const baseColor = hexToRgb(coverColor);

      for (let i = 0; i < data.length; i += 4) {
        const noise = Math.random() * 40 - 20;
        data[i] = Math.max(0, Math.min(255, baseColor.r + noise));
        data[i + 1] = Math.max(0, Math.min(255, baseColor.g + noise));
        data[i + 2] = Math.max(0, Math.min(255, baseColor.b + noise));
        data[i + 3] = 255;
      }

      ctx.putImageData(imageData, 0, 0);
    }

    const dataUrl = canvas.toDataURL('image/png');
    setGeneratedCover(dataUrl);

    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 600,
      useNativeDriver: true,
    }).start();
  };

  const saveCover = () => {
    if (!generatedCover) return;

    if (Platform.OS === 'web') {
      const link = document.createElement('a');
      link.href = generatedCover;
      link.download = `cover_${width}x${height}_${coverPattern}.png`;
      link.click();
    }
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#1a0b2e', '#2d1b4e', '#16213e', '#0f3460']}
        style={StyleSheet.absoluteFill}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      />

      <View style={[styles.safeArea, { paddingTop: insets.top }]}>
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.header}>
            <Wrench size={32} color="#a78bfa" />
            <Text style={styles.title}>Tools</Text>
            <Text style={styles.subtitle}>Utilities & Generators</Text>
          </View>

          <View style={styles.toolCard}>
            <View style={styles.toolHeader}>
              <Calculator size={24} color="#a78bfa" />
              <Text style={styles.toolTitle}>Capacity Calculator</Text>
            </View>
            <Text style={styles.toolDescription}>
              Calculate the maximum payload capacity for a given image size
            </Text>

            <View style={styles.inputRow}>
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Width (px)</Text>
                <TextInput
                  style={styles.input}
                  placeholder="1024"
                  placeholderTextColor="#6b7280"
                  keyboardType="numeric"
                  value={width}
                  onChangeText={setWidth}
                />
              </View>
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Height (px)</Text>
                <TextInput
                  style={styles.input}
                  placeholder="768"
                  placeholderTextColor="#6b7280"
                  keyboardType="numeric"
                  value={height}
                  onChangeText={setHeight}
                />
              </View>
            </View>

            <Pressable style={styles.calculateButton} onPress={calculateCapacity}>
              <Calculator size={18} color="#fff" />
              <Text style={styles.calculateButtonText}>Calculate</Text>
            </Pressable>

            {capacityResult && (
              <Animated.View style={[styles.resultCard, { opacity: fadeAnim }]}>
                <Text style={styles.resultTitle}>Capacity Analysis</Text>
                <View style={styles.resultRow}>
                  <Text style={styles.resultLabel}>Total Pixels:</Text>
                  <Text style={styles.resultValue}>{capacityResult.totalPixels.toLocaleString()}</Text>
                </View>
                <View style={styles.resultRow}>
                  <Text style={styles.resultLabel}>Total Bits:</Text>
                  <Text style={styles.resultValue}>{capacityResult.totalBits.toLocaleString()}</Text>
                </View>
                <View style={styles.resultRow}>
                  <Text style={styles.resultLabel}>Max Payload:</Text>
                  <Text style={styles.resultValue}>{capacityResult.maxPayloadBytes.toLocaleString()} bytes</Text>
                </View>
                <View style={styles.resultRow}>
                  <Text style={styles.resultLabel}>Max Characters:</Text>
                  <Text style={styles.resultValue}>{capacityResult.maxPayloadChars.toLocaleString()}</Text>
                </View>
              </Animated.View>
            )}
          </View>

          <View style={styles.toolCard}>
            <View style={styles.toolHeader}>
              <Palette size={24} color="#a78bfa" />
              <Text style={styles.toolTitle}>Cover Generator</Text>
            </View>
            <Text style={styles.toolDescription}>
              Generate cover images for steganography
            </Text>

            <View style={styles.patternSelector}>
              <Text style={styles.inputLabel}>Pattern</Text>
              <View style={styles.patternButtons}>
                {(['solid', 'gradient', 'noise'] as const).map((pattern) => (
                  <Pressable
                    key={pattern}
                    style={[
                      styles.patternButton,
                      coverPattern === pattern && styles.patternButtonActive,
                    ]}
                    onPress={() => setCoverPattern(pattern)}
                  >
                    <Text
                      style={[
                        styles.patternButtonText,
                        coverPattern === pattern && styles.patternButtonTextActive,
                      ]}
                    >
                      {pattern.charAt(0).toUpperCase() + pattern.slice(1)}
                    </Text>
                  </Pressable>
                ))}
              </View>
            </View>

            <View style={styles.colorSelector}>
              <Text style={styles.inputLabel}>Base Color</Text>
              <View style={styles.colorRow}>
                <TextInput
                  style={styles.colorInput}
                  placeholder="#1a0b2e"
                  placeholderTextColor="#6b7280"
                  value={coverColor}
                  onChangeText={setCoverColor}
                />
                <View style={[styles.colorPreview, { backgroundColor: coverColor }]} />
              </View>
            </View>

            <Pressable style={styles.generateButton} onPress={generateCover}>
              <Palette size={18} color="#fff" />
              <Text style={styles.generateButtonText}>Generate Cover</Text>
            </Pressable>

            {generatedCover && (
              <Animated.View style={[styles.coverPreviewCard, { opacity: fadeAnim }]}>
                <Text style={styles.resultTitle}>Generated Cover</Text>
                <Animated.Image
                  source={{ uri: generatedCover }}
                  style={styles.coverPreview}
                  resizeMode="contain"
                />
                <Pressable style={styles.saveCoverButton} onPress={saveCover}>
                  <Text style={styles.saveCoverButtonText}>Save Cover Image</Text>
                </Pressable>
              </Animated.View>
            )}
          </View>
        </ScrollView>
      </View>
    </View>
  );
}

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : { r: 26, g: 11, b: 46 };
}

function adjustBrightness(hex: string, percent: number): string {
  const rgb = hexToRgb(hex);
  const adjust = (val: number) => Math.max(0, Math.min(255, val + percent));
  return `rgb(${adjust(rgb.r)}, ${adjust(rgb.g)}, ${adjust(rgb.b)})`;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    alignItems: 'center' as const,
    marginBottom: 32,
    marginTop: 20,
  },
  title: {
    fontSize: 36,
    fontWeight: '700' as const,
    color: '#e9d5ff',
    marginTop: 12,
    letterSpacing: 1,
  },
  subtitle: {
    fontSize: 14,
    color: '#a78bfa',
    marginTop: 4,
    letterSpacing: 2,
    textTransform: 'uppercase' as const,
  },
  toolCard: {
    backgroundColor: 'rgba(167, 139, 250, 0.05)',
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(167, 139, 250, 0.2)',
    marginBottom: 24,
  },
  toolHeader: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    gap: 12,
    marginBottom: 8,
  },
  toolTitle: {
    fontSize: 22,
    fontWeight: '700' as const,
    color: '#e9d5ff',
  },
  toolDescription: {
    fontSize: 14,
    color: '#a78bfa',
    marginBottom: 20,
    lineHeight: 20,
  },
  inputRow: {
    flexDirection: 'row' as const,
    gap: 12,
    marginBottom: 16,
  },
  inputGroup: {
    flex: 1,
  },
  inputLabel: {
    fontSize: 14,
    color: '#a78bfa',
    marginBottom: 8,
    fontWeight: '600' as const,
  },
  input: {
    backgroundColor: 'rgba(167, 139, 250, 0.1)',
    borderRadius: 12,
    padding: 14,
    color: '#e9d5ff',
    fontSize: 15,
    borderWidth: 1,
    borderColor: 'rgba(167, 139, 250, 0.2)',
  },
  calculateButton: {
    backgroundColor: '#a78bfa',
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 20,
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    gap: 8,
  },
  calculateButtonText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '700' as const,
  },
  resultCard: {
    backgroundColor: 'rgba(34, 197, 94, 0.1)',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    borderWidth: 1,
    borderColor: 'rgba(34, 197, 94, 0.3)',
  },
  resultTitle: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: '#e9d5ff',
    marginBottom: 12,
  },
  resultRow: {
    flexDirection: 'row' as const,
    justifyContent: 'space-between' as const,
    marginBottom: 8,
  },
  resultLabel: {
    fontSize: 14,
    color: '#a78bfa',
  },
  resultValue: {
    fontSize: 14,
    color: '#e9d5ff',
    fontWeight: '600' as const,
  },
  patternSelector: {
    marginBottom: 16,
  },
  patternButtons: {
    flexDirection: 'row' as const,
    gap: 8,
  },
  patternButton: {
    flex: 1,
    backgroundColor: 'rgba(167, 139, 250, 0.1)',
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: 'rgba(167, 139, 250, 0.2)',
    alignItems: 'center' as const,
  },
  patternButtonActive: {
    backgroundColor: 'rgba(167, 139, 250, 0.3)',
    borderColor: '#a78bfa',
  },
  patternButtonText: {
    fontSize: 13,
    color: '#a78bfa',
    fontWeight: '600' as const,
  },
  patternButtonTextActive: {
    color: '#e9d5ff',
  },
  colorSelector: {
    marginBottom: 16,
  },
  colorRow: {
    flexDirection: 'row' as const,
    gap: 12,
    alignItems: 'center' as const,
  },
  colorInput: {
    flex: 1,
    backgroundColor: 'rgba(167, 139, 250, 0.1)',
    borderRadius: 12,
    padding: 14,
    color: '#e9d5ff',
    fontSize: 15,
    borderWidth: 1,
    borderColor: 'rgba(167, 139, 250, 0.2)',
  },
  colorPreview: {
    width: 48,
    height: 48,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: 'rgba(167, 139, 250, 0.4)',
  },
  generateButton: {
    backgroundColor: '#a78bfa',
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 20,
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    gap: 8,
  },
  generateButtonText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '700' as const,
  },
  coverPreviewCard: {
    backgroundColor: 'rgba(167, 139, 250, 0.1)',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    borderWidth: 1,
    borderColor: 'rgba(167, 139, 250, 0.2)',
  },
  coverPreview: {
    width: '100%',
    height: 200,
    borderRadius: 12,
    marginBottom: 12,
  },
  saveCoverButton: {
    backgroundColor: 'rgba(167, 139, 250, 0.3)',
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 20,
    alignItems: 'center' as const,
  },
  saveCoverButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600' as const,
  },
});
