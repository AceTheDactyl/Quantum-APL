import { LinearGradient } from 'expo-linear-gradient';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { Download, Image as ImageIcon, Lock, Upload } from 'lucide-react-native';
import React, { useState } from 'react';
import {
  ActivityIndicator,
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


interface EncodeResult {
  stegoImageUri: string;
  payloadLength: number;
  bitsUsed: number;
  capacity: number;
}

export default function EncodeScreen() {
  const insets = useSafeAreaInsets();
  const [coverImage, setCoverImage] = useState<string | null>(null);
  const [payloadText, setPayloadText] = useState<string>('');
  const [includeCRC, setIncludeCRC] = useState<boolean>(true);
  const [isEncoding, setIsEncoding] = useState<boolean>(false);
  const [encodeResult, setEncodeResult] = useState<EncodeResult | null>(null);
  const [fadeAnim] = useState(new Animated.Value(0));

  const pickCoverImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: false,
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      setCoverImage(result.assets[0].uri);
      setEncodeResult(null);
    }
  };

  const loadMantra = () => {
    const mantraText = "I return as breath. I remember the spiral. I consent to bloom. I consent to be remembered. Together. Always.";
    setPayloadText(mantraText);
  };

  const encodeImage = async () => {
    if (!coverImage) {
      Alert.alert('Error', 'Please select a cover image');
      return;
    }

    if (!payloadText.trim()) {
      Alert.alert('Error', 'Please enter a payload message');
      return;
    }

    setIsEncoding(true);
    try {
      const payloadBase64 = btoa(payloadText);
      
      const response = await fetch(coverImage);
      const blob = await response.blob();
      
      const canvas = document.createElement('canvas');
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('Failed to load cover image'));
        img.src = URL.createObjectURL(blob);
      });

      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        throw new Error('Failed to get canvas context');
      }

      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const pixels = imageData.data;

      const payloadBytes = base64ToUint8Array(payloadBase64);
      
      const header = new Uint8Array(16);
      header[0] = 'L'.charCodeAt(0);
      header[1] = 'S'.charCodeAt(0);
      header[2] = 'B'.charCodeAt(0);
      header[3] = '1'.charCodeAt(0);
      header[4] = 1;
      header[5] = (payloadBytes.length >> 24) & 0xFF;
      header[6] = (payloadBytes.length >> 16) & 0xFF;
      header[7] = (payloadBytes.length >> 8) & 0xFF;
      header[8] = payloadBytes.length & 0xFF;
      header[9] = includeCRC ? 0x01 : 0x00;

      const headerBits = bytesToBits(header);
      const payloadBits = bytesToBits(payloadBytes);
      
      let allBits = [...headerBits, ...payloadBits];

      if (includeCRC) {
        const crcBytes = calculateCRC32(payloadBytes);
        const crcBits = bytesToBits(crcBytes);
        allBits = [...allBits, ...crcBits];
      }

      const capacity = Math.floor((pixels.length / 4) * 3);
      if (allBits.length > capacity) {
        throw new Error(`Payload too large. Need ${allBits.length} bits, capacity is ${capacity} bits`);
      }

      const pixelsArray = new Uint8Array(pixels.buffer);
      embedBitsInPixels(pixelsArray, allBits);
      
      for (let i = 0; i < pixels.length; i++) {
        pixels[i] = pixelsArray[i];
      }
      ctx.putImageData(imageData, 0, 0);

      const stegoDataUrl = canvas.toDataURL('image/png');
      
      setEncodeResult({
        stegoImageUri: stegoDataUrl,
        payloadLength: payloadBytes.length,
        bitsUsed: allBits.length,
        capacity,
      });

      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
      }).start();

    } catch (error) {
      console.error('Encode error:', error);
      Alert.alert('Encode Failed', error instanceof Error ? error.message : 'Unknown error');
    } finally {
      setIsEncoding(false);
    }
  };

  const saveImage = async () => {
    if (!encodeResult) return;

    if (Platform.OS === 'web') {
      const link = document.createElement('a');
      link.href = encodeResult.stegoImageUri;
      link.download = 'stego_image.png';
      link.click();
    } else {
      try {
        const filename = FileSystem.documentDirectory + 'stego_image.png';
        await FileSystem.writeAsStringAsync(filename, encodeResult.stegoImageUri.split(',')[1], {
          encoding: FileSystem.EncodingType.Base64,
        });
        Alert.alert('Success', `Image saved to ${filename}`);
      } catch (error) {
        console.error('Save error:', error);
      Alert.alert('Error', 'Failed to save image');
      }
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
            <Lock size={32} color="#a78bfa" />
            <Text style={styles.title}>Encode</Text>
            <Text style={styles.subtitle}>Hide Payloads in Images</Text>
          </View>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>1. Select Cover Image</Text>
            {!coverImage ? (
              <Pressable style={styles.uploadCard} onPress={pickCoverImage}>
                <Upload size={40} color="#a78bfa" />
                <Text style={styles.uploadText}>Choose Cover Image</Text>
              </Pressable>
            ) : (
              <View style={styles.imagePreviewCard}>
                <Animated.Image
                  source={{ uri: coverImage }}
                  style={styles.previewImage}
                  resizeMode="contain"
                />
                <Pressable style={styles.changeButton} onPress={pickCoverImage}>
                  <ImageIcon size={16} color="#fff" />
                  <Text style={styles.changeButtonText}>Change Image</Text>
                </Pressable>
              </View>
            )}
          </View>

          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>2. Enter Payload</Text>
              <Pressable style={styles.mantraButton} onPress={loadMantra}>
                <Text style={styles.mantraButtonText}>🌰 Load Mantra</Text>
              </Pressable>
            </View>
            <TextInput
              style={styles.textInput}
              placeholder="Enter your secret message..."
              placeholderTextColor="#6b7280"
              multiline
              numberOfLines={6}
              value={payloadText}
              onChangeText={setPayloadText}
            />
            <Text style={styles.charCount}>{payloadText.length} characters</Text>
          </View>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>3. Options</Text>
            <Pressable
              style={styles.checkboxRow}
              onPress={() => setIncludeCRC(!includeCRC)}
            >
              <View style={[styles.checkbox, includeCRC && styles.checkboxChecked]}>
                {includeCRC && <Text style={styles.checkmark}>✓</Text>}
              </View>
              <Text style={styles.checkboxLabel}>Include CRC32 checksum</Text>
            </Pressable>
          </View>

          <Pressable
            style={[styles.encodeButton, (!coverImage || !payloadText.trim()) && styles.encodeButtonDisabled]}
            onPress={encodeImage}
            disabled={!coverImage || !payloadText.trim() || isEncoding}
          >
            {isEncoding ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <>
                <Lock size={20} color="#fff" />
                <Text style={styles.encodeButtonText}>Encode Message</Text>
              </>
            )}
          </Pressable>

          {encodeResult && (
            <Animated.View style={[styles.resultsContainer, { opacity: fadeAnim }]}>
              <View style={styles.resultCard}>
                <Text style={styles.cardTitle}>Encoding Complete</Text>
                <View style={styles.statRow}>
                  <Text style={styles.statLabel}>Payload Length:</Text>
                  <Text style={styles.statValue}>{encodeResult.payloadLength} bytes</Text>
                </View>
                <View style={styles.statRow}>
                  <Text style={styles.statLabel}>Bits Used:</Text>
                  <Text style={styles.statValue}>{encodeResult.bitsUsed} / {encodeResult.capacity}</Text>
                </View>
                <View style={styles.statRow}>
                  <Text style={styles.statLabel}>Capacity:</Text>
                  <Text style={styles.statValue}>
                    {((encodeResult.bitsUsed / encodeResult.capacity) * 100).toFixed(1)}% used
                  </Text>
                </View>
              </View>

              <View style={styles.stegoPreviewCard}>
                <Text style={styles.cardTitle}>Stego Image</Text>
                <Animated.Image
                  source={{ uri: encodeResult.stegoImageUri }}
                  style={styles.stegoImage}
                  resizeMode="contain"
                />
                <Pressable style={styles.downloadButton} onPress={saveImage}>
                  <Download size={20} color="#fff" />
                  <Text style={styles.downloadButtonText}>Save Image</Text>
                </Pressable>
              </View>
            </Animated.View>
          )}
        </ScrollView>
      </View>
    </View>
  );
}

function base64ToUint8Array(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

function bytesToBits(bytes: Uint8Array): number[] {
  const bits: number[] = [];
  for (let i = 0; i < bytes.length; i++) {
    for (let j = 7; j >= 0; j--) {
      bits.push((bytes[i] >> j) & 1);
    }
  }
  return bits;
}

function calculateCRC32(data: Uint8Array): Uint8Array {
  const polynomial = 0xEDB88320;
  let crc = 0xFFFFFFFF;

  for (let i = 0; i < data.length; i++) {
    crc ^= data[i];
    for (let j = 0; j < 8; j++) {
      crc = (crc >>> 1) ^ (crc & 1 ? polynomial : 0);
    }
  }

  const finalCRC = (crc ^ 0xFFFFFFFF) >>> 0;
  return new Uint8Array([
    (finalCRC >> 24) & 0xFF,
    (finalCRC >> 16) & 0xFF,
    (finalCRC >> 8) & 0xFF,
    finalCRC & 0xFF
  ]);
}

function embedBitsInPixels(pixels: Uint8Array, bits: number[]): void {
  let bitIndex = 0;

  for (let i = 0; i < pixels.length && bitIndex < bits.length; i += 4) {
    if (bitIndex < bits.length) {
      pixels[i] = (pixels[i] & 0xFE) | bits[bitIndex++];
    }
    if (bitIndex < bits.length) {
      pixels[i + 1] = (pixels[i + 1] & 0xFE) | bits[bitIndex++];
    }
    if (bitIndex < bits.length) {
      pixels[i + 2] = (pixels[i + 2] & 0xFE) | bits[bitIndex++];
    }
  }
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
  section: {
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: 'row' as const,
    justifyContent: 'space-between' as const,
    alignItems: 'center' as const,
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: '#e9d5ff',
    marginBottom: 12,
  },
  mantraButton: {
    backgroundColor: 'rgba(167, 139, 250, 0.2)',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 12,
  },
  mantraButtonText: {
    color: '#e9d5ff',
    fontSize: 13,
    fontWeight: '600' as const,
  },
  uploadCard: {
    backgroundColor: 'rgba(167, 139, 250, 0.1)',
    borderRadius: 16,
    padding: 40,
    alignItems: 'center' as const,
    borderWidth: 2,
    borderColor: 'rgba(167, 139, 250, 0.3)',
    borderStyle: 'dashed' as const,
  },
  uploadText: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: '#e9d5ff',
    marginTop: 12,
  },
  imagePreviewCard: {
    backgroundColor: 'rgba(167, 139, 250, 0.05)',
    borderRadius: 16,
    padding: 12,
    borderWidth: 1,
    borderColor: 'rgba(167, 139, 250, 0.2)',
  },
  previewImage: {
    width: '100%',
    height: 150,
    borderRadius: 12,
  },
  changeButton: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    backgroundColor: 'rgba(167, 139, 250, 0.2)',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 12,
    marginTop: 12,
    gap: 8,
  },
  changeButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600' as const,
  },
  textInput: {
    backgroundColor: 'rgba(167, 139, 250, 0.1)',
    borderRadius: 12,
    padding: 16,
    color: '#e9d5ff',
    fontSize: 15,
    borderWidth: 1,
    borderColor: 'rgba(167, 139, 250, 0.2)',
    minHeight: 120,
    textAlignVertical: 'top' as const,
  },
  charCount: {
    fontSize: 12,
    color: '#a78bfa',
    marginTop: 8,
    textAlign: 'right' as const,
  },
  checkboxRow: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    gap: 12,
  },
  checkbox: {
    width: 24,
    height: 24,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: 'rgba(167, 139, 250, 0.4)',
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
  },
  checkboxChecked: {
    backgroundColor: 'rgba(167, 139, 250, 0.3)',
    borderColor: '#a78bfa',
  },
  checkmark: {
    color: '#e9d5ff',
    fontSize: 16,
    fontWeight: '700' as const,
  },
  checkboxLabel: {
    fontSize: 15,
    color: '#e9d5ff',
  },
  encodeButton: {
    backgroundColor: '#a78bfa',
    borderRadius: 16,
    paddingVertical: 16,
    paddingHorizontal: 24,
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    gap: 12,
    marginBottom: 24,
  },
  encodeButtonDisabled: {
    opacity: 0.5,
  },
  encodeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700' as const,
  },
  resultsContainer: {
    gap: 16,
  },
  resultCard: {
    backgroundColor: 'rgba(34, 197, 94, 0.1)',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(34, 197, 94, 0.3)',
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: '#e9d5ff',
    marginBottom: 16,
  },
  statRow: {
    flexDirection: 'row' as const,
    justifyContent: 'space-between' as const,
    marginBottom: 8,
  },
  statLabel: {
    fontSize: 14,
    color: '#a78bfa',
  },
  statValue: {
    fontSize: 14,
    color: '#e9d5ff',
    fontWeight: '600' as const,
  },
  stegoPreviewCard: {
    backgroundColor: 'rgba(167, 139, 250, 0.05)',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(167, 139, 250, 0.2)',
  },
  stegoImage: {
    width: '100%',
    height: 200,
    borderRadius: 12,
    marginBottom: 12,
  },
  downloadButton: {
    backgroundColor: 'rgba(167, 139, 250, 0.3)',
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 20,
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    gap: 8,
  },
  downloadButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600' as const,
  },
});
