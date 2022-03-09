import tensorflow as tf

def tf_lower_and_split_punct(text):
  # Split accecented characters.
  text = tf.strings.lower(text)
  # Keep space, a to z, and select punctuation.
  text = tf.strings.regex_replace(text, '[^ a-z.?!,():¿]', '')
  # Add spaces around punctuation.
  text = tf.strings.regex_replace(text, '[.?!,():¿]', r' \0 ')
  # Strip whitespace.
  text = tf.strings.strip(text)

  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
  return text
    