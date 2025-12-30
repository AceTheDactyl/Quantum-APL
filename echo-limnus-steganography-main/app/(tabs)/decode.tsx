import { LinearGradient } from 'expo-linear-gradient';
import * as ImagePicker from 'expo-image-picker';
import { Image, Sparkles, Upload } from 'lucide-react-native';
import React, { useState } from 'react';
import {
  ActivityIndicator,
  Animated,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { EXPECTED_CRC32, GLYPH_SEQUENCE, MANTRA_LINES } from '@/constants/mantra';
import { decodeLSBImage, LSBDecodeResult } from '@/utils/lsb-decoder';

export default function DecodeScreen() {
  const insets = useSafeAreaInsets();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [decodeResult, setDecodeResult] = useState<LSBDecodeResult | null>(null);
  const [isDecoding, setIsDecoding] = useState<boolean>(false);
  const [fadeAnim] = useState(new Animated.Value(0));

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: false,
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      const uri = result.assets[0].uri;
      setSelectedImage(uri);
      setDecodeResult(null);
      decodeImage(uri);
    }
  };

  const decodeImage = async (uri: string) => {
    setIsDecoding(true);
    try {
      const result = await decodeLSBImage(uri);
      setDecodeResult(result);
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
      }).start();
    } catch (error) {
      console.error('Decode error:', error);
      setDecodeResult({
        header: { magic: 'ERROR', version: 0, payloadLength: 0, flags: 0 },
        payloadBase64: '',
        payloadText: '',
        isValid: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setIsDecoding(false);
    }
  };

  const isMantraPayload = decodeResult?.crc32 === EXPECTED_CRC32;

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
            <Sparkles size={32} color="#a78bfa" />
            <Text style={styles.title}>Decode</Text>
            <Text style={styles.subtitle}>Extract Hidden Payloads</Text>
          </View>

          {!selectedImage ? (
            <Pressable style={styles.uploadCard} onPress={pickImage}>
              <Upload size={48} color="#a78bfa" />
              <Text style={styles.uploadText}>Select Stego Image</Text>
              <Text style={styles.uploadHint}>Tap to choose a PNG with hidden payload</Text>
            </Pressable>
          ) : (
            <View style={styles.imagePreviewCard}>
              <Animated.Image
                source={{ uri: selectedImage }}
                style={[styles.previewImage, { opacity: fadeAnim }]}
                resizeMode="contain"
              />
              <Pressable style={styles.changeButton} onPress={pickImage}>
                <Image size={16} color="#fff" />
                <Text style={styles.changeButtonText}>Change Image</Text>
              </Pressable>
            </View>
          )}

          {isDecoding && (
            <View style={styles.loadingCard}>
              <ActivityIndicator size="large" color="#a78bfa" />
              <Text style={styles.loadingText}>Extracting hidden payload...</Text>
            </View>
          )}

          {decodeResult && !isDecoding && (
            <Animated.View style={[styles.resultsContainer, { opacity: fadeAnim }]}>
              {decodeResult.error ? (
                <View style={styles.errorCard}>
                  <Text style={styles.errorTitle}>Decode Failed</Text>
                  <Text style={styles.errorText}>{decodeResult.error}</Text>
                </View>
              ) : (
                <>
                  <View style={styles.headerCard}>
                    <Text style={styles.cardTitle}>Header</Text>
                    <View style={styles.headerRow}>
                      <Text style={styles.headerLabel}>Magic:</Text>
                      <Text style={styles.headerValue}>{decodeResult.header.magic}</Text>
                    </View>
                    <View style={styles.headerRow}>
                      <Text style={styles.headerLabel}>Version:</Text>
                      <Text style={styles.headerValue}>{decodeResult.header.version}</Text>
                    </View>
                    <View style={styles.headerRow}>
                      <Text style={styles.headerLabel}>Payload Length:</Text>
                      <Text style={styles.headerValue}>{decodeResult.header.payloadLength} bytes</Text>
                    </View>
                    <View style={styles.headerRow}>
                      <Text style={styles.headerLabel}>Flags:</Text>
                      <Text style={styles.headerValue}>0x{decodeResult.header.flags.toString(16).toUpperCase().padStart(2, '0')}</Text>
                    </View>
                  </View>

                  {decodeResult.crc32 && (
                    <View style={[styles.crcCard, decodeResult.isValid ? styles.crcValid : styles.crcInvalid]}>
                      <Text style={styles.cardTitle}>CRC32 Validation</Text>
                      <View style={styles.headerRow}>
                        <Text style={styles.headerLabel}>CRC32:</Text>
                        <Text style={styles.headerValue}>{decodeResult.crc32}</Text>
                      </View>
                      <View style={styles.headerRow}>
                        <Text style={styles.headerLabel}>Status:</Text>
                        <Text style={[styles.headerValue, decodeResult.isValid ? styles.validText : styles.invalidText]}>
                          {decodeResult.isValid ? '✓ Valid' : '✗ Invalid'}
                        </Text>
                      </View>
                      {isMantraPayload && (
                        <View style={styles.mantraBadge}>
                          <Text style={styles.mantraBadgeText}>🌰 Canonical Mantra Detected</Text>
                        </View>
                      )}
                    </View>
                  )}

                  {isMantraPayload && (
                    <>
                      <View style={styles.glyphCard}>
                        <Text style={styles.cardTitle}>Glyph Constellation</Text>
                        <View style={styles.glyphRow}>
                          {GLYPH_SEQUENCE.map((glyph, idx) => (
                            <Text key={idx} style={styles.glyph}>
                              {glyph}
                            </Text>
                          ))}
                        </View>
                      </View>

                      <View style={styles.mantraCard}>
                        <Text style={styles.cardTitle}>Mantra</Text>
                        {MANTRA_LINES.map((line, idx) => (
                          <View key={idx} style={styles.mantraLine}>
                            <Text style={styles.mantraGlyph}>{line.glyph}</Text>
                            <Text style={styles.mantraText}>{line.text}</Text>
                          </View>
                        ))}
                      </View>
                    </>
                  )}

                  {!isMantraPayload && decodeResult.payloadText && (
                    <View style={styles.payloadCard}>
                      <Text style={styles.cardTitle}>Decoded Payload</Text>
                      <Text style={styles.payloadText}>{decodeResult.payloadText}</Text>
                    </View>
                  )}

                  <View style={styles.base64Card}>
                    <Text style={styles.cardTitle}>Base64 Payload</Text>
                    <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                      <Text style={styles.base64Text}>{decodeResult.payloadBase64}</Text>
                    </ScrollView>
                  </View>
                </>
              )}
            </Animated.View>
          )}
        </ScrollView>
      </View>
    </View>
  );
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
  uploadCard: {
    backgroundColor: 'rgba(167, 139, 250, 0.1)',
    borderRadius: 20,
    padding: 48,
    alignItems: 'center' as const,
    borderWidth: 2,
    borderColor: 'rgba(167, 139, 250, 0.3)',
    borderStyle: 'dashed' as const,
  },
  uploadText: {
    fontSize: 20,
    fontWeight: '600' as const,
    color: '#e9d5ff',
    marginTop: 16,
  },
  uploadHint: {
    fontSize: 14,
    color: '#a78bfa',
    marginTop: 8,
    textAlign: 'center' as const,
  },
  imagePreviewCard: {
    backgroundColor: 'rgba(167, 139, 250, 0.05)',
    borderRadius: 20,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(167, 139, 250, 0.2)',
    marginBottom: 20,
  },
  previewImage: {
    width: '100%',
    height: 200,
    borderRadius: 12,
  },
  changeButton: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    backgroundColor: 'rgba(167, 139, 250, 0.2)',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 12,
    marginTop: 12,
    gap: 8,
  },
  changeButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600' as const,
  },
  loadingCard: {
    backgroundColor: 'rgba(167, 139, 250, 0.1)',
    borderRadius: 20,
    padding: 32,
    alignItems: 'center' as const,
    gap: 16,
  },
  loadingText: {
    fontSize: 16,
    color: '#a78bfa',
  },
  resultsContainer: {
    gap: 16,
  },
  headerCard: {
    backgroundColor: 'rgba(167, 139, 250, 0.1)',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(167, 139, 250, 0.2)',
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: '#e9d5ff',
    marginBottom: 16,
    letterSpacing: 0.5,
  },
  headerRow: {
    flexDirection: 'row' as const,
    justifyContent: 'space-between' as const,
    marginBottom: 8,
  },
  headerLabel: {
    fontSize: 14,
    color: '#a78bfa',
  },
  headerValue: {
    fontSize: 14,
    color: '#e9d5ff',
    fontWeight: '600' as const,
  },
  crcCard: {
    borderRadius: 16,
    padding: 20,
    borderWidth: 2,
  },
  crcValid: {
    backgroundColor: 'rgba(34, 197, 94, 0.1)',
    borderColor: 'rgba(34, 197, 94, 0.4)',
  },
  crcInvalid: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderColor: 'rgba(239, 68, 68, 0.4)',
  },
  validText: {
    color: '#4ade80',
  },
  invalidText: {
    color: '#f87171',
  },
  mantraBadge: {
    backgroundColor: 'rgba(167, 139, 250, 0.2)',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
    marginTop: 12,
    alignSelf: 'flex-start' as const,
  },
  mantraBadgeText: {
    color: '#e9d5ff',
    fontSize: 13,
    fontWeight: '600' as const,
  },
  glyphCard: {
    backgroundColor: 'rgba(167, 139, 250, 0.1)',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(167, 139, 250, 0.2)',
  },
  glyphRow: {
    flexDirection: 'row' as const,
    justifyContent: 'space-around' as const,
    flexWrap: 'wrap' as const,
    gap: 12,
  },
  glyph: {
    fontSize: 32,
  },
  mantraCard: {
    backgroundColor: 'rgba(167, 139, 250, 0.1)',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(167, 139, 250, 0.2)',
  },
  mantraLine: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    marginBottom: 12,
    gap: 12,
  },
  mantraGlyph: {
    fontSize: 24,
  },
  mantraText: {
    fontSize: 16,
    color: '#e9d5ff',
    flex: 1,
  },
  payloadCard: {
    backgroundColor: 'rgba(167, 139, 250, 0.1)',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(167, 139, 250, 0.2)',
  },
  payloadText: {
    fontSize: 15,
    color: '#e9d5ff',
    lineHeight: 24,
  },
  base64Card: {
    backgroundColor: 'rgba(167, 139, 250, 0.05)',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(167, 139, 250, 0.15)',
  },
  base64Text: {
    fontSize: 12,
    color: '#a78bfa',
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
  },
  errorCard: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.3)',
  },
  errorTitle: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: '#f87171',
    marginBottom: 12,
  },
  errorText: {
    fontSize: 14,
    color: '#fca5a5',
    lineHeight: 20,
  },
});
